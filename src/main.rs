use std::{
    collections::VecDeque,
    f32::{consts::PI, EPSILON},
    iter,
};

use bevy::{
    ecs::{schedule::ScheduleLabel, system::EntityCommands},
    input::mouse::{MouseButtonInput, MouseMotion},
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    utils::{hashbrown::HashSet, HashMap},
    window::PrimaryWindow,
};
use bitflags::bitflags;
use itertools::Itertools;
use nalgebra::{OPoint, Point2, SVector};
use parry2d::{na::Isometry2, shape::TriMesh};
use rand::thread_rng;

bitflags! {
    #[derive(PartialEq, Eq, Hash, Clone, Copy, Default, Debug)]
    pub struct CollisionFlags: u32 {
        const OBJECT = 0b00000001;
        const PARTICLE = 0b00000010;
    }
}

bitflags! {
    #[derive(PartialEq, Eq, Hash, Clone, Copy, Default, Debug)]
    pub struct Direction: u32 {
        const LEFT = 0b00000001;
        const RIGHT = 0b00000010;
        const DOWN = 0b00000100;
        const UP = 0b00001000;
    }
}

impl Direction {
    fn as_vec(self) -> Vec2 {
        let mut direction = Vec2::ZERO;

        if self & Self::LEFT != Self::empty() {
            direction -= Vec2::X;
        }
        if self & Self::RIGHT != Self::empty() {
            direction += Vec2::X;
        }
        if self & Self::DOWN != Self::empty() {
            direction -= Vec2::Y;
        }
        if self & Self::UP != Self::empty() {
            direction += Vec2::Y;
        }

        direction.normalize_or_zero()
    }
}

#[derive(Component)]
pub enum Particle {
    Bullet,
}

#[derive(Component)]
pub struct Destruct(f32, f32);

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum Cell {
    Metal,
    Stone,
    Thruster(Direction),
    Gun(Direction),
}

impl Cell {
    fn color(&self) -> [f32; 4] {
        match self {
            Self::Metal => [0.8, 0.8, 0.8, 1.0],
            Self::Stone => [0.4, 0.4, 0.4, 1.0],
            Self::Thruster(_) => [0.0, 0.0, 0.8, 1.0],
            Self::Gun(_) => [1.0, 0.0, 0.0, 1.0],
        }
    }
    fn mass(&self) -> f32 {
        match self {
            Self::Metal => 10.0,
            Self::Stone => 100.0,
            Self::Gun(_) => 10.0,
            Self::Thruster(_) => 0.4,
        }
    }
}

#[derive(Component, Default)]
pub struct Object {
    data: HashMap<IVec2, Cell>,
    minimum: IVec2,
    maximum: IVec2,
}

impl Object {
    fn calc_bounds(&mut self) {
        let mut minimum = IVec2::MAX;
        let mut inc_maximum = IVec2::MIN;

        for (pos, _) in &self.data {
            if minimum.x > pos.x {
                minimum.x = pos.x;
            }
            if minimum.y > pos.y {
                minimum.y = pos.y;
            }
            if inc_maximum.x < pos.x {
                inc_maximum.x = pos.x;
            }
            if inc_maximum.y < pos.y {
                inc_maximum.y = pos.y;
            }
        }

        self.minimum = minimum;
        self.maximum = inc_maximum + 1;
    }
    fn size(&self) -> UVec2 {
        (self.maximum - self.minimum).as_uvec2()
    }
    fn total_mass(&self) -> f32 {
        self.data.iter().map(|(_, c)| c.mass()).sum::<f32>()
    }
    fn center_of_mass(&self) -> Vec2 {
        let mut pos_accum = Vec2::ZERO;
        let mut mass_accum = 0.0;
        for (pos, cell) in &self.data {
            pos_accum += cell.mass() * pos.as_vec2();
            mass_accum += cell.mass();
        }
        pos_accum / mass_accum
    }
}

const TEMPLATE_VERTICES: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
];

const TEMPLATE_INDICES: [u32; 6] = [0, 3, 1, 1, 3, 2];

fn create_object_mesh(object: &Object) -> MeshData {
    let mut visited = HashSet::new();

    let mut groups = HashMap::<Cell, Vec<(IVec2, IVec2)>>::new();

    let mut cursor = object.minimum;

    loop {
        let mut size = IVec2::splat(0);

        'a: for x in object.minimum.x..object.maximum.x {
            for y in object.minimum.y..object.maximum.y {
                if !visited.contains(&IVec2 { x, y }) && object.data.contains_key(&IVec2 { x, y }) {
                    cursor = IVec2 { x, y };
                    break 'a;
                }
            }
        }

        let template = object.data[&cursor].clone();
        while object.data.contains_key(&(cursor + size))
            && object.data[&(cursor + size)] == template
        {
            size.x += 1;
        }
        size.y += 1;

        loop {
            let range = (cursor.x..cursor.x + size.x).map(move |x| IVec2::new(x, size.y));
            if range.clone().all(|offset| {
                object.data.contains_key(&(cursor + offset))
                    && object.data[&(cursor + offset)] == template
            }) {
                size.y += 1;
            } else {
                break;
            }
        }

        for x in cursor.x..cursor.x + size.x {
            for y in cursor.y..cursor.y + size.y {
                if object.data.contains_key(&IVec2 { x, y }) {
                    visited.insert(IVec2 { x, y });
                }
            }
        }

        groups.entry(template).or_default().push((cursor, size));

        if visited.len() == object.data.len() {
            break;
        }
    }

    let mut vertices = vec![];
    let mut indices = vec![];
    let mut colors = vec![];

    for (cell, pairs) in groups {
        for (pos, size) in pairs {
            let count = vertices.len();
            vertices.extend(TEMPLATE_VERTICES.iter().copied().map(|[mut x, mut y, z]| {
                x = x * size.x as f32 + pos.x as f32;
                y = y * size.y as f32 + pos.y as f32;
                [x, y, z]
            }));
            indices.extend(TEMPLATE_INDICES.iter().copied().map(|i| i + count as u32));
            colors.extend(iter::repeat(cell.color()).take(TEMPLATE_VERTICES.len()));
        }
    }

    MeshData {
        vertices,
        indices,
        colors,
    }
}

pub struct MeshData {
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
    colors: Vec<[f32; 4]>,
}

fn spawn_object(
    commands: &mut Commands,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    meshes: &mut ResMut<Assets<Mesh>>,
    object: Object,
    transform: Transform,
) -> Entity {
    commands
        .spawn(ObjectBundle::from_object(
            object, transform, materials, meshes,
        ))
        .id()
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut camera_transform = Transform::from_xyz(0.0, 0.0, 100.0);
    camera_transform.look_at(Vec3::ZERO, Vec3::Y);
    let camera_id = commands
        .spawn(Camera2dBundle {
            transform: camera_transform,
            projection: OrthographicProjection {
                scale: 0.1,
                ..default()
            },
            ..default()
        })
        .id();

    let mut data = HashMap::new();
    data.insert(IVec2::new(0, 1), Cell::Metal);
    data.insert(IVec2::new(1, 1), Cell::Metal);
    data.insert(IVec2::new(2, 1), Cell::Metal);
    data.insert(IVec2::new(1, 2), Cell::Gun(Direction::UP));
    data.insert(IVec2::new(0, 0), Cell::Thruster(Direction::UP));
    data.insert(IVec2::new(2, 0), Cell::Thruster(Direction::UP));
    data.insert(IVec2::new(0, 2), Cell::Thruster(Direction::DOWN));
    data.insert(IVec2::new(2, 2), Cell::Thruster(Direction::DOWN));
    data.insert(IVec2::new(-1, 1), Cell::Thruster(Direction::LEFT));
    data.insert(IVec2::new(3, 1), Cell::Thruster(Direction::RIGHT));
    let mut object = Object { data, ..default() };
    object.calc_bounds();

    let ship_entity = spawn_object(
        &mut commands,
        &mut materials,
        &mut meshes,
        object,
        Transform::from_xyz(0.0, 0.0, 0.0),
    );

    commands
        .entity(ship_entity)
        .insert(Input::default())
        .add_child(camera_id);

    fn generate_asteroid_object() -> Object {
        use noise::*;
        use rand::Rng;

        let mut rng = thread_rng();

        let noise = Fbm::<Perlin>::new(rng.gen());

        let radius = rng.gen_range(10..20);
        let x2_radius = 2 * radius;

        let mut object = Object::default();

        for x in -x2_radius..=x2_radius {
            for y in -x2_radius..=x2_radius {
                let diff = radius as f32 - IVec2 { x, y }.as_vec2().length();
                const SQUISH: f32 = 0.3;

                let density_mod = SQUISH * diff;

                let density = noise.get([x as f64 * 0.1, y as f64 * 0.1]) as f32;

                if density + density_mod > 0.0 {
                    object.data.insert(IVec2 { x, y }, Cell::Stone);
                }
            }
        }
        object.calc_bounds();
        object
    }

    use rand::Rng;

    let mut rng = thread_rng();

    for i in 0..10 {
        let asteroid_id = spawn_object(
            &mut commands,
            &mut materials,
            &mut meshes,
            generate_asteroid_object(),
            Transform::from_xyz(
                400.0 * rng.gen::<f32>() - 200.0,
                400.0 * rng.gen::<f32>() - 200.0,
                0.0,
            ),
        );
    }
}

fn shoot_guns(
    mut commands: Commands,
    mut query: Query<(&Input, &GlobalTransform, &LinearVelocity, &Object)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query1: Query<(&Parent, &Camera)>,
) {
    let (p, _) = query1.single();

    for (i, t, lv, o) in query.iter() {
        let r = t.to_scale_rotation_translation().1;
        let mut available_guns = HashMap::new();

        for x in o.minimum.x..o.maximum.x {
            for y in o.minimum.y..o.maximum.y {
                let Some(Cell::Gun(direction)) = o.data.get(&IVec2 { x, y }) else {
                    continue;
                };

                available_guns.insert(IVec2 { x, y }, direction);
            }
        }

        available_guns.retain(|_, dir| {
            let direction = dir.as_vec();

            direction.dot(i.fire) > 0.7
        });

        let material = materials.add(ColorMaterial {
            color: Color::Rgba {
                red: 1.0,
                green: 1.0,
                blue: 0.0,
                alpha: 1.0,
            },
            ..default()
        });

        for (pos, dir) in &available_guns {
            let mesh = meshes.add(
                Mesh::new(PrimitiveTopology::TriangleList)
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        TEMPLATE_VERTICES.map(|[a,b,c]| [0.5 * a, b, c]).to_vec().clone(),
                    )
                    //.with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
                    .with_indices(Some(Indices::U32(TEMPLATE_INDICES.to_vec().clone()))),
            );

            let global = (r * (pos.as_vec2() + dir.as_vec() * 1.5).extend(0.))
                .xy() + t.translation().xy();
            dbg!(lv.0);
            let collider = Collider(TriMesh::new(
                TEMPLATE_VERTICES
                    .into_iter()
                    .map(|[x, y, _]| Point2::new(x, y))
                    .collect::<Vec<_>>(),
                TEMPLATE_INDICES
                    .chunks(3)
                    .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect::<Vec<_>>(),
            ));
            let mut transform = Transform::from_xyz(global.x, global.y, 0.0);
            transform.rotate_z(r.to_euler(EulerRot::XYZ).2);
            commands.spawn(ParticleBundle {
                particle: Particle::Bullet,
                physics: PhysicsBundle {
                    pos: Pos(global + TICK * lv.0 + 0.05 * (r * i.fire.extend(0.)).xy()),
                    prev_pos: PrevPos(global),
                    rot: Rot::from_radians(r.to_euler(EulerRot::XYZ).2),
                    prev_rot: PrevRot(Rot::from_radians(r.to_euler(EulerRot::XYZ).2)),
                    mass: Mass(0.01),
                    size: Size(UVec2::splat(1)),
                    collider,
                    flags: CollisionFlag(CollisionFlags::PARTICLE),
                    ..default()
                },
                destruct: Destruct(0.0, 1.1),
                material: MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(mesh),
                    material: material.clone(),
                    transform,
                    ..default()
                },
            });
        }
    }
}

fn solve_particles(query: Query<(&Particle, &Pos)>, mut meshes: ResMut<Assets<Mesh>>) {
    for (particle, mesh_handle) in query.iter() {
        dbg!(mesh_handle.0);
    }
}

#[derive(Component, Default, Debug)]
pub struct Input {
    direction: Vec2,
    rotate: f32,
    fire: Vec2,
}

fn input(
    query1: Query<(&Parent, &Camera)>,
    mut query2: Query<&mut Input>,
    keys: Res<bevy::prelude::Input<KeyCode>>,
    mouse: Res<bevy::prelude::Input<MouseButton>>,
    window: Query<&Window, With<PrimaryWindow>>,
    axes: Res<Axis<GamepadAxis>>,
    buttons: Res<bevy::prelude::Input<GamepadButton>>,
    gamepads: Option<Res<Gamepads>>,
) {
    let _ = keys;
    let (p, _) = query1.single();

    let mut i = query2.get_mut(p.get()).unwrap();

    // if let Some(gamepads) = gamepads {
    //     for gamepad in gamepads.iter() {
    //         let axis_lx = GamepadAxis {
    //             gamepad, axis_type: GamepadAxisType::LeftStickX
    //         };
    //         let axis_ly = GamepadAxis {
    //             gamepad, axis_type: GamepadAxisType::LeftStickY
    //         };
    //         if let (Some(x), Some(y)) = (axes.get(axis_lx), axes.get(axis_ly)) {
    //             let left_stick_pos = Vec2::new(x, y);

    //             if left_stick_pos.length() > 0.05 {
    //                 i.direction = left_stick_pos;
    //             } else {
    //                 i.direction = Vec2::ZERO;
    //             }
    //         }
    //         let left_bumper = GamepadButton {
    //             gamepad, button_type: GamepadButtonType::LeftTrigger
    //         };
    //         let right_bumper = GamepadButton {
    //             gamepad, button_type: GamepadButtonType::RightTrigger
    //         };
    //         i.rotate = if buttons.pressed(left_bumper) {
    //             1.0
    //         } else {
    //             0.0
    //         } + if buttons.pressed(right_bumper) {
    //             -1.0
    //         } else {
    //             0.0
    //         };
    //     }
    // } else {
    let mut d = Direction::default();
    if keys.pressed(KeyCode::W) {
        d |= Direction::UP;
    }
    if keys.pressed(KeyCode::S) {
        d |= Direction::DOWN;
    }
    if keys.pressed(KeyCode::A) {
        d |= Direction::LEFT;
    }
    if keys.pressed(KeyCode::D) {
        d |= Direction::RIGHT;
    }
    i.direction = d.as_vec();

    i.rotate = keys.pressed(KeyCode::Q) as i32 as f32 - keys.pressed(KeyCode::E) as i32 as f32;

    let window = window.single();

    if mouse.pressed(MouseButton::Left) {
        i.fire = (Vec2::new(window.resolution.width(), window.resolution.height()) / 2.0
            - window.cursor_position().unwrap())
        .normalize_or_zero()
            * Vec2::new(-1.0, 1.0);
    } else {
        i.fire = Vec2::ZERO;
    }

    // }
}
#[derive(Component, Clone, Copy)]
pub struct Rot {
    cos: f32,
    sin: f32,
}

impl Default for Rot {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Rot {
    pub const ZERO: Self = Self { cos: 1., sin: 0. };
    pub fn from_radians(radians: f32) -> Self {
        Self {
            cos: radians.cos(),
            sin: radians.sin(),
        }
    }

    pub fn from_degrees(degrees: f32) -> Self {
        let radians = degrees.to_radians();
        Self::from_radians(radians)
    }

    pub fn as_radians(&self) -> f32 {
        f32::atan2(self.sin, self.cos)
    }
    pub fn rotate(&self, vec: Vec2) -> Vec2 {
        Vec2::new(
            vec.x * self.cos - vec.y * self.sin,
            vec.x * self.sin + vec.y * self.cos,
        )
    }
    pub fn inv(self) -> Self {
        Self {
            cos: self.cos,
            sin: -self.sin,
        }
    }
    pub fn mul(self, rhs: Rot) -> Self {
        Self {
            cos: self.cos * rhs.cos - self.sin * rhs.sin,
            sin: self.sin * rhs.cos + self.cos * rhs.sin,
        }
    }
}

#[derive(Component, Default)]
pub struct PrevRot(Rot);
#[derive(Component, Default)]
pub struct CenterOfMass(Vec2);

#[derive(Component, Default)]
pub struct CollisionFlag(CollisionFlags);
#[derive(Component)]
pub struct Mass(f32);
impl Default for Mass {
    fn default() -> Self {
        Self(1.0)
    }
}
#[derive(Component, Default)]
pub struct LinearVelocity(Vec2);
#[derive(Component, Default)]
pub struct AngularVelocity(f32);
#[derive(Component, Default)]
pub struct ExternalForce(Vec2);
#[derive(Component, Default)]
pub struct ExternalTorque(f32);
#[derive(Component, Default)]
pub struct Pos(Vec2);
#[derive(Component, Default)]
pub struct PrevPos(Vec2);

#[derive(Component)]
pub struct Collider(TriMesh);

#[derive(Component, Default)]
pub struct Size(UVec2);

impl Default for Collider {
    fn default() -> Self {
        Self(TriMesh::new(
            iter::repeat(Point2::<f32>::default())
                .take(3)
                .collect::<Vec<_>>(),
            vec![[0u32, 1, 2]],
        ))
    }
}

pub const SUBSTEPS: usize = 10;
pub const DELTA: f32 = 1.0 / 100.0;
pub const TICK: f32 = DELTA / SUBSTEPS as f32;

fn integrate(
    mut query: Query<(
        &mut Pos,
        &mut PrevPos,
        &mut Rot,
        &mut PrevRot,
        &mut LinearVelocity,
        &mut AngularVelocity,
        &Mass,
        &Size,
        &ExternalForce,
        &ExternalTorque,
    )>,
) {
    for (
        mut pos,
        mut prev_pos,
        mut rot,
        mut prev_rot,
        mut linear_velocity,
        mut angular_velocity,
        mass,
        size,
        force,
        torque,
    ) in query.iter_mut()
    {
        {
            let acceleration = mass.0 * force.0;
            linear_velocity.0 += TICK * acceleration / mass.0;
        }
        {
            let acceleration = torque.0;
            let moment_of_inertia = mass.0 * size.0.as_vec2().length_squared();
            angular_velocity.0 += TICK * acceleration / moment_of_inertia;
        }

        pos.0 += TICK * linear_velocity.0;
        *rot = Rot::from_radians(rot.as_radians() + TICK * angular_velocity.0);
    }
}

fn solve_positions(
    mut query: Query<(
        Entity,
        &mut Rot,
        &mut Pos,
        &Size,
        &Collider,
        &Mass,
        &CollisionFlag,
    )>,
) {
    let mut iter = query.iter_combinations_mut();
    while let Some(
        [(e_a, mut rot_a, mut pos_a, size_a, collider_a, mass_a, flag_a), (e_b, mut rot_b, mut pos_b, size_b, collider_b, mass_b, flag_b)],
    ) = iter.fetch_next()
    {
        if flag_a.0 & flag_b.0 == CollisionFlags::PARTICLE {
            continue;
        }
        let iso_a = Isometry2::new(
            SVector::<f32, 2>::new(pos_a.0.x, pos_a.0.y),
            rot_a.as_radians(),
        );
        let iso_b = Isometry2::new(
            SVector::<f32, 2>::new(pos_b.0.x, pos_b.0.y),
            rot_b.as_radians(),
        );

        let Ok(true) =
            parry2d::query::intersection_test(&iso_a, &collider_a.0, &iso_b, &collider_b.0)
        else {
            continue;
        };

        let Ok(Some(contact)) =
            parry2d::query::contact::contact(&iso_a, &collider_a.0, &iso_b, &collider_b.0, 10.0)
        else {
            continue;
        };

        let parry2d::query::contact::Contact {
            point1,
            point2,
            normal1,
            normal2,
            dist,
        } = contact;

        let m_a = 1.0 / mass_a.0;
        let m_b = 1.0 / mass_b.0;

        let inertia_a_inv = m_a / size_a.0.as_vec2().length_squared();
        let inertia_b_inv = m_b / size_b.0.as_vec2().length_squared();

        let w_a_rot = inertia_a_inv * point1.coords.perp(&normal1).powi(2);
        let w_b_rot = inertia_b_inv * point2.coords.perp(&normal2).powi(2);

        let m_a_r = m_a + w_a_rot;
        let m_b_r = m_b + w_b_rot;
        let m_sum = m_a_r + m_b_r;

        let impulse_a = Vec2::new(normal1[0], normal1[1]) * dist * m_a_r / m_sum;
        let impulse_b = Vec2::new(normal2[0], normal2[1]) * dist * m_b_r / m_sum;
        pos_a.0 += impulse_a;
        pos_b.0 += impulse_b;

        *rot_a = rot_a.mul(Rot::from_radians(
            inertia_a_inv
                * point1
                    .coords
                    .perp(&SVector::<f32, 2>::new(impulse_a.x, impulse_a.y)),
        ));
        *rot_b = rot_b.mul(Rot::from_radians(
            inertia_b_inv
                * point2
                    .coords
                    .perp(&SVector::<f32, 2>::new(impulse_b.x, impulse_b.y)),
        ));
    }
}

fn solve_velocities() {}

fn update_velocities(
    mut query: Query<(
        &mut Pos,
        &mut PrevPos,
        &mut Rot,
        &mut PrevRot,
        &mut LinearVelocity,
        &mut AngularVelocity,
    )>,
) {
    for (mut pos, mut prev_pos, mut rot, mut prev_rot, mut linear_velocity, mut angular_velocity) in
        query.iter_mut()
    {
        linear_velocity.0 = (pos.0 - prev_pos.0) / TICK;
        angular_velocity.0 = (rot.as_radians() - prev_rot.0.as_radians()) / TICK;
        prev_pos.0 = pos.0;
        prev_rot.0 = *rot;
    }
}

fn sync_transforms(mut query: Query<(&mut Transform, &Rot, &Pos)>) {
    for (mut transform, rot, pos) in query.iter_mut() {
        transform.translation = pos.0.extend(0.);
        transform.rotation = Quat::from_rotation_z(rot.as_radians());
    }
}

fn collect_collision_pairs() {}

fn run_physics(world: &mut World) {
    world.run_schedule(Physics);
}

#[derive(Debug, Default)]
pub struct PhysicsPlugin;

#[derive(SystemSet, Hash, Debug, PartialEq, Eq, Clone, Copy)]
enum PhysicsSet {
    CollectCollisionPairs,
    Integrate,
    SolvePositions,
    UpdateVelocities,
    SolveVelocities,
}

#[derive(ScheduleLabel, Hash, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Physics;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        use PhysicsSet::*;
        app.add_systems(
            Physics,
            (collect_collision_pairs).in_set(CollectCollisionPairs),
        )
        .add_systems(Physics, (integrate).in_set(Integrate))
        .add_systems(Physics, (solve_positions).in_set(SolvePositions))
        .add_systems(Physics, (update_velocities).in_set(UpdateVelocities))
        .add_systems(Physics, (solve_velocities).in_set(SolveVelocities))
        .configure_sets(Physics, Integrate.after(CollectCollisionPairs))
        .configure_sets(Physics, SolvePositions.after(Integrate))
        .configure_sets(Physics, UpdateVelocities.after(SolvePositions))
        .configure_sets(Physics, SolveVelocities.after(UpdateVelocities))
        .add_systems(FixedUpdate, run_physics)
        .add_systems(Update, sync_transforms);
    }
}

pub fn destruct(time: Res<Time>, mut q: Query<(Entity, &mut Destruct)>, mut commands: Commands) {
    for (e, mut d) in q.iter_mut() {
        d.0 += time.delta_seconds();
        if d.0 >= d.1 {
            commands.entity(e).despawn();
        }
    }
}

/*
&Object,
       &Input,
       &GlobalTransform,
       &CenterOfMass,
       &Mass,
       &mut LinearVelocity,
       &mut AngularVelocity,
       &mut ExternalForce,
       &mut ExternalTorque, */

#[derive(Bundle, Default)]
pub struct PhysicsBundle {
    pub pos: Pos,
    pub prev_pos: PrevPos,
    pub rot: Rot,
    pub prev_rot: PrevRot,
    pub size: Size,
    pub center_of_mass: CenterOfMass,
    pub mass: Mass,
    pub linear_velocity: LinearVelocity,
    pub angular_velocity: AngularVelocity,
    pub external_force: ExternalForce,
    pub external_torque: ExternalTorque,
    pub collider: Collider,
    pub flags: CollisionFlag,
}
#[derive(Bundle, Default)]
pub struct ObjectBundle {
    pub object: Object,
    pub physics: PhysicsBundle,
    pub material: MaterialMesh2dBundle<ColorMaterial>,
}
#[derive(Bundle)]
pub struct ParticleBundle {
    pub particle: Particle,
    pub physics: PhysicsBundle,
    pub material: MaterialMesh2dBundle<ColorMaterial>,
    pub destruct: Destruct,
}

impl ObjectBundle {
    fn from_object(
        object: Object,
        transform: Transform,
        materials: &mut ResMut<Assets<ColorMaterial>>,
        meshes: &mut ResMut<Assets<Mesh>>,
    ) -> Self {
        let MeshData {
            vertices,
            indices,
            colors,
        } = create_object_mesh(&object);

        let mesh = meshes.add(
            Mesh::new(PrimitiveTopology::TriangleList)
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices.clone())
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
                .with_indices(Some(Indices::U32(indices.clone()))),
        );

        let material = materials.add(ColorMaterial {
            color: Color::Rgba {
                red: 1.0,
                green: 1.0,
                blue: 1.0,
                alpha: 1.0,
            },
            ..default()
        });
        let collider = Collider(TriMesh::new(
            vertices
                .into_iter()
                .map(|[x, y, _]| Point2::new(x, y))
                .collect::<Vec<_>>(),
            indices
                .chunks(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect::<Vec<_>>(),
        ));
        Self {
            physics: PhysicsBundle {
                center_of_mass: CenterOfMass(object.center_of_mass()),
                size: Size(object.size()),
                mass: Mass(object.total_mass()),
                pos: Pos(transform.translation.xy()),
                prev_pos: PrevPos(transform.translation.xy()),
                flags: CollisionFlag(CollisionFlags::OBJECT),
                collider,
                ..default()
            },
            material: MaterialMesh2dBundle {
                mesh: Mesh2dHandle(mesh),
                material: material,
                transform,
                ..default()
            },
            object,
            ..default()
        }
    }
}

fn thruster_alloc(
    mut query: Query<(
        &Object,
        &Input,
        &GlobalTransform,
        &CenterOfMass,
        &Mass,
        &mut LinearVelocity,
        &mut AngularVelocity,
        &mut ExternalForce,
        &mut ExternalTorque,
    )>,
) {
    for (o, i, t, com, m, mut v, mut a, mut ef, mut et) in query.iter_mut() {
        let r = t.to_scale_rotation_translation().1;

        let mut available_thrusters = HashMap::new();

        for x in o.minimum.x..o.maximum.x {
            for y in o.minimum.y..o.maximum.y {
                let Some(Cell::Thruster(direction)) = o.data.get(&IVec2 { x, y }) else {
                    continue;
                };

                available_thrusters.insert(IVec2 { x, y }, direction);
            }
        }

        let mut desired_torque = 0.0;
        let mut desired_force = Vec2::default();

        let mut should_thrust = false;

        let v_mag = v.0.length();

        let d_mag = i.direction.length();

        const MAX_V_MAG: f32 = 90.0;
        const MAX_A: f32 = PI / 2.0;
        const F: f32 = 10000.0;
        const TF: f32 = 1200.0;

        if v_mag > MAX_V_MAG * d_mag {
            v.0 = MAX_V_MAG * d_mag * v.0.normalize_or_zero();
        }
        if a.0.abs() > MAX_A * i.rotate.abs() {
            a.0 = MAX_A * i.rotate;
        }

        if i.rotate != 0.0 {
            desired_torque = F * i.rotate;
            should_thrust = true;
        }

        if i.direction != Vec2::default() {
            desired_force = F * (r * (i.direction).extend(0.)).xy();
            should_thrust = true;
        }

        if !should_thrust {}

        if should_thrust {
            let mut thrusters_in_combo = vec![];
            for k in 0..available_thrusters.len() {
                for combos in available_thrusters.iter().combinations(k) {
                    let mut v = vec![];
                    for (pos, direction) in combos {
                        let force_scaling;
                        if desired_force != Vec2::ZERO
                            && desired_torque == 0.0
                            && i.direction != Vec2::ZERO
                        {
                            force_scaling = i.direction.dot(direction.as_vec());
                        } else {
                            force_scaling = i.rotate.abs().clamp(0.0, 1.0);
                        }
                        let aa = pos.as_vec2() + 0.5 - com.0;
                        let bb = force_scaling * (r * direction.as_vec().extend(0.)).xy();

                        let torque = aa.x * bb.y - aa.y * bb.x;
                        v.push((*pos, ExternalForce(bb), ExternalTorque(torque)));
                    }
                    thrusters_in_combo.push(v);
                }
            }

            let mut weighted_outcomes = HashMap::new();
            let mut index = None;

            {
                for (a, thrusters) in thrusters_in_combo.iter().enumerate() {
                    let mut force = Vec2::ZERO;
                    let mut torque = 0.0f32;
                    for thruster in thrusters {
                        force += thruster.1 .0;
                        torque += thruster.2 .0;
                    }
                    if force.length() == 0.0f32 && torque == 0.0f32 {
                        continue;
                    }
                    force = force.normalize_or_zero();
                    weighted_outcomes.insert(
                        a,
                        (
                            desired_force.dot(force),
                            (desired_torque - torque).abs(),
                            force,
                        ),
                    );
                }
                let mut lowest_weight = f32::MAX;
                let mut highest_weight = f32::MIN;
                for (a, weight) in &weighted_outcomes {
                    if weight.0 >= highest_weight && weight.1 <= lowest_weight {
                        index = Some(*a);
                        highest_weight = highest_weight.max(weight.0);
                        lowest_weight = lowest_weight.min(weight.1);
                    }
                }
            }

            if let Some(i) = index {
                let mut force = ExternalForce(Vec2::ZERO);
                let mut torque = ExternalTorque(0.0);
                for thruster in &thrusters_in_combo[i] {
                    let aa = thruster.0.as_vec2() + 0.5 - com.0;
                    let bb = TF * thruster.1 .0;

                    torque.0 += aa.x * bb.y - aa.y * bb.x;
                    force.0 += bb;
                }
                *ef = force;
                *et = torque;
            }
        }
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, PhysicsPlugin));
    app.add_systems(Startup, setup);
    app.add_systems(Update, thruster_alloc);
    app.add_systems(Update, input);
    app.add_systems(Update, shoot_guns);
    app.add_systems(Update, solve_particles);
    app.add_systems(Update, destruct);
    app.insert_resource(Time::<Fixed>::from_seconds(TICK as f64));
    app.run();
}
