use std::{
    f32::{consts::PI, EPSILON},
    iter,
};

use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    utils::{hashbrown::{HashSet}, HashMap}, ecs::system::EntityCommands,
};
use bevy_xpbd_2d::{parry::na::ComplexField, prelude::*};
use bitflags::bitflags;
use itertools::Itertools;
use rand::thread_rng;

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

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum Cell {
    Metal,
    Stone,
    Thruster(Direction),
}

impl Cell {
    fn color(&self) -> [f32; 4] {
        match self {
            Self::Metal => [0.8, 0.8, 0.8, 1.0],
            Self::Stone => [0.4, 0.4, 0.4, 1.0],
            Self::Thruster(_) => [0.0, 0.0, 0.8, 1.0],
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
}

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

    const TEMPLATE_VERTICES: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ];

    const TEMPLATE_INDICES: [u32; 6] = [0, 3, 1, 1, 3, 2];

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
    meshes: &mut ResMut<Assets<Mesh>>, object: Object, transform: Transform) -> Entity {
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
    let collider = Collider::trimesh(
        vertices
            .into_iter()
            .map(|[x, y, _]| Vec2 { x, y })
            .collect::<Vec<_>>(),
        indices
            .chunks(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>(),
    );
    commands
        .spawn((
            Input::default(),
            RigidBody::Dynamic,
            object,
            collider.clone(),
            MaterialMesh2dBundle {
                mesh: Mesh2dHandle(mesh.clone()),
                material: material.clone(),
                transform,
                ..default()
            },
        )).id()
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
                scale: 0.15,
                ..default()
            },
            ..default()
        })
        .id();

    let mut data = HashMap::new();
    data.insert(IVec2::new(0, 1), Cell::Metal);
    data.insert(IVec2::new(1, 1), Cell::Metal);
    data.insert(IVec2::new(2, 1), Cell::Metal);
    data.insert(IVec2::new(0, 0), Cell::Thruster(Direction::UP));
    data.insert(IVec2::new(2, 0), Cell::Thruster(Direction::UP));
    data.insert(IVec2::new(0, 2), Cell::Thruster(Direction::DOWN));
    data.insert(IVec2::new(2, 2), Cell::Thruster(Direction::DOWN));
    data.insert(IVec2::new(-1, 1), Cell::Thruster(Direction::LEFT));
    data.insert(IVec2::new(3, 1), Cell::Thruster(Direction::RIGHT));
    let mut object = Object { data, ..default() };
    object.calc_bounds();

    let ship_entity = spawn_object(&mut commands, &mut materials, &mut meshes, object, Transform::from_xyz(0.0, 0.0, 0.0));

    commands.entity(ship_entity).add_child(camera_id);
    
    fn generate_asteroid_object() -> Object {
        use rand::Rng;
        use noise::*;

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

    for i in 0..20 {
        use rand::Rng;

        let mut rng = thread_rng();

        let asteroid_id = spawn_object(&mut commands, &mut materials, &mut meshes, generate_asteroid_object(), Transform::from_xyz(400.0 * rng.gen::<f32>() - 200.0, 400.0 * rng.gen::<f32>() - 200.0, 0.0));
    }

}

#[derive(Component, Default, Debug)]
pub struct Input {
    direction: Vec2,
    rotate: f32,
}

fn input(
    query1: Query<(&Parent, &Camera)>,
    mut query2: Query<&mut Input>,
    keys: Res<bevy::prelude::Input<KeyCode>>,
) {
    let _ = keys;
    let (p, _) = query1.single();

    let mut i = query2.get_mut(p.get()).unwrap();

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
}

fn thruster_alloc(
    mut query: Query<(
        &Object,
        &Input,
        &GlobalTransform,
        &CenterOfMass,
        &mut LinearVelocity,
        &mut AngularVelocity,
        &mut ExternalForce,
        &mut ExternalTorque,
    )>,
) {
    for (o, i, t, com, mut v, mut a, mut ef, mut et) in query.iter_mut() {
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

        if !should_thrust {
            
        }

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
                        let mut force = ExternalForce::default();
                        force.apply_force_at_point(
                            force_scaling * (r * direction.as_vec().extend(0.)).xy(),
                            pos.as_vec2() + 0.5,
                            com.0,
                        );

                        v.push((*pos, force));
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
                        force += thruster.1.force();
                        torque += thruster.1.torque();
                    }
                    if force.length() == 0.0f32 && torque == 0.0f32 {
                        continue;
                    }
                    force = force.normalize_or_zero();
                    weighted_outcomes.insert(a, (desired_force.dot(force), (desired_torque - torque).abs(), force));
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
                let mut force = ExternalForce::default().with_persistence(false);
                for thruster in &thrusters_in_combo[i] {
                    force.apply_force_at_point(TF * thruster.1.force(), thruster.0.as_vec2() + 0.5, com.0);
                }
                *ef = force;
            }
        }
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, PhysicsPlugins::default()));
    app.insert_resource(Gravity(Vec2::splat(0.0)));
    app.add_systems(Startup, setup);
    app.add_systems(Update, thruster_alloc);
    app.add_systems(Update, input);
    app.run();
}
