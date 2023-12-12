use std::{iter, f32::{consts::PI, EPSILON}};

use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    utils::{hashbrown::HashSet, HashMap}, sprite::{Mesh2dHandle, MaterialMesh2dBundle},
};
use bevy_xpbd_2d::{prelude::*, parry::na::ComplexField};
use bitflags::bitflags;
use itertools::Itertools;

bitflags! {
    #[derive(PartialEq, Eq, Hash, Clone, Copy, Default)]
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
    Thruster(Direction),
}

impl Cell {
    fn color(&self) -> [f32; 4] {
        match self {
            Self::Metal => [0.8, 0.8, 0.8, 1.0],
            Self::Thruster(_) => [0.0, 0.0, 0.8, 1.0],
        }
    }
}

#[derive(Component)]
pub struct Object {
    data: HashMap<IVec2, Cell>,
    minimum: IVec2,
    maximum: IVec2,
}

impl Object {
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

        for x in object.minimum.x..object.maximum.x {
            for y in object.minimum.y..object.maximum.y {
                if !visited.contains(&IVec2 { x, y }) && object.data.contains_key(&IVec2 { x, y }) {
                    cursor = IVec2 { x, y };
                }
            }
        }
        

        let template = object.data[&cursor].clone();
        while object.data.contains_key(&(cursor + size))
            && object.data[&(cursor + size)] == template
        {
            visited.insert(cursor + size);
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
                for offset in range {
                    visited.insert(cursor + offset);
                }
            } else {
                break;
            }
        }

            groups.entry(template).or_default().push((cursor, size));
        cursor = cursor + size;

        
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
        
        }}

    MeshData { vertices, indices, colors }
}

pub struct MeshData {
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
    colors: Vec<[f32; 4]>,
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut camera_transform = Transform::from_xyz(0.0, 0.0, 100.0);
    camera_transform.look_at(Vec3::ZERO, Vec3::Y);
    camera_transform.rotate_z(PI);
    let camera_id = commands
        .spawn(Camera2dBundle { transform: camera_transform, projection: OrthographicProjection { scale: 0.2, ..default()}, ..default() }).id();
    

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
    let object = Object { data, minimum: IVec2::new(-1, 0), maximum: IVec2::new(4, 3) };

    let MeshData { vertices, indices, colors } = create_object_mesh(&object);

    let mesh = meshes.add(Mesh::new(PrimitiveTopology::TriangleList)
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_indices(Some(Indices::U32(indices.clone()))));

    let material = materials.add(ColorMaterial {
        color: Color::Rgba {
            red: 1.0,
            green: 1.0,
            blue: 1.0,
            alpha: 1.0,
        },
        ..default()
    });
    let collider = Collider::trimesh(vertices.into_iter().map(|[x, y, _]| Vec2 { x, y} ).collect::<Vec<_>>(), indices.chunks(3).map(|chunk| [chunk[0], chunk[1], chunk[2]]).collect::<Vec<_>>());
    commands.spawn((
        Input::default(),
        RigidBody::Dynamic,
        ExternalForce::default(),
        ExternalTorque::default(),
        LinearVelocity::default(),
        AngularVelocity::default(),
        object,
        collider.clone(),
        MaterialMesh2dBundle {
            mesh: Mesh2dHandle(mesh.clone()),
            material: material.clone(),
            transform: Transform::from_xyz(10.0, 0.0, 0.0),
            ..default()
        },
    )).add_child(camera_id);
    commands.spawn((
        RigidBody::Dynamic,
        collider,
        MaterialMesh2dBundle {
            mesh: Mesh2dHandle(mesh),
            material,
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..default()
        },
    ));
}

#[derive(Component, Default, Debug)]
pub struct Input {
    direction: Vec2,
    rotate: f32,
}

fn input(query1: Query<(&Parent, &Camera)>, mut query2: Query<&mut Input>, keys: Res<bevy::prelude::Input<KeyCode>>,) {
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

fn thruster_alloc(mut query: Query<(&Object, &Input, &Transform, &CenterOfMass, &mut LinearVelocity, &mut AngularVelocity, &mut ExternalForce, &mut ExternalTorque)>) {
    for (o, i, t, com, mut v, mut a, mut ef, mut et) in query.iter_mut() {
        let rn = t.rotation.to_euler(EulerRot::ZYX);
        let mut r = Mat2::default();
        r.x_axis = Vec2::new(rn.0.cos(), rn.0.sin());
        r.y_axis = Vec2::new(-rn.0.sin(), rn.0.cos());


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

        const MAX_V_MAG: f32 = 50.0;
        const MAX_A: f32 = PI / 2.0;
        const R: f32 = 0.99;
        const F: f32 = 10000.0;
        const TF: f32 = 1200.0;

        if v_mag > MAX_V_MAG * d_mag {
            v.0 = MAX_V_MAG * d_mag * v.0.normalize_or_zero();
        }
        if a.0.abs() > MAX_A * i.rotate.abs() {
            a.0 = MAX_A * i.rotate;
        } else 
            if v_mag >= 0.0 {
                a.0 = R * a.0;
        }

        if i.rotate != 0.0 {
            desired_torque = F * i.rotate;
            should_thrust = true;
        }

        if i.direction != Vec2::default() {
            desired_force = F * i.direction;
            should_thrust = true;
        }

        if !should_thrust {
            let rev_t = F * -a.0.signum();
            let rev_f = F * -(r.inverse() * v.0);

            let inertial_dampeners_on = false;

            if a.0.abs() > EPSILON {
                desired_torque = rev_t;
                should_thrust = true;
            } else {
                a.0 = default();
            }

            if desired_torque == 0.0 && v.0.length() > EPSILON {
                desired_force = rev_f;
                should_thrust = true;
            } else {
                v.0 = default();
            }
        }

        if should_thrust {
            let mut thrusters_in_combo = vec![];
            for k in 0..available_thrusters.len() {
                for combos in available_thrusters.iter().combinations(k) {
                    let mut v = vec![];
                    for (pos, direction) in combos {
                        let mut force_scaling = 0.0;
                        if desired_force != Vec2::ZERO && desired_torque == 0.0 && i.direction != Vec2::ZERO {
                            force_scaling = i.direction.length().clamp(0.0, 1.0);
                        } else {
                            force_scaling = i.rotate.abs().clamp(0.0, 1.0);
                        }
                        let mut force = ExternalForce::default();
                        force.apply_force_at_point(force_scaling * TF * direction.as_vec(), pos.as_vec2() + 0.5, com.0);
                        v.push((*pos, force,));
                    }
                    thrusters_in_combo.push(v);
                }
            }

            let mut weighted_outcomes = HashMap::new();
            let mut index = None;

            {
                let mut force = Vec2::ZERO;
                let mut torque = 0.0f32;
                for (a, thrusters) in thrusters_in_combo.iter().enumerate() {
                    for thruster in thrusters {
                        force += thruster.1.force();
                        torque += thruster.1.torque();
                    }
                    weighted_outcomes.insert(a, desired_force.dot(force) + (desired_torque - torque).abs());
                }
                dbg!(desired_force.dot(force), (desired_torque - torque).abs());
                let mut lowest_weight = f32::MAX;
                for (a, weight) in weighted_outcomes {
                    if weight < lowest_weight {
                        index = Some(a);
                        lowest_weight = weight;
                    }
                }
            }

            if let Some(i) = index {
                let mut force = ExternalForce::default();
                for thruster in &thrusters_in_combo[i] {
                    force.apply_force_at_point(thruster.1.force(), thruster.0.as_vec2() + 0.5, com.0);
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