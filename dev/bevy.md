# Bevy

- https://bevyengine.org/
- https://github.com/bevyengine/bevy
- coordinate system
    - y-up, right handed
- written in [Rust](rust.md).

## TODO

- take a look at the rendering crate

## Getting started

## Glossary

- `App`
    - the main container
    - (fields)
        - sub_apps
        - runner: RunnerFn
    - (methods)
        - register_required_components()
- `AppExit`
    - enum
    - variants
        - `Success`
        - `Error`
- `Arc`
    - stands for "Automatically Reference Counted"
- `Archetype`
    - a kind of type associated with a bundle (a set of components)
    - (fields)
        - `id: ArchetypeId<u32>`
        - `table_id: TableId<u32>`
        - `edges: Edges`
        - `entities: Vec<ArchetypeEntity>`
        - `components: ImmutableSparseSet<ComponentId, ArchetypeComponentInfo>`
        - `flags: ArchetypeFlags`
- `ArchetypeGeneration`
- `Archetypes`
    - https://taintedcoders.com/bevy/archetypes
    - (fields)
        - `archetypes: Vec<Archetype>`
        - `by_components: HashMap<ArchetypeComponents, ArchetypeId>`
        - `by_component: ComponentIndex`
    - (methods)
        - ...
        - `generation(&self) -> ArchetypeGeneration`
- `Asset`
    - supposed to be loaded asynchronously
- `Assets`
- `AssetServer`
    - (methods)
        - `load() -> Handle<Asset>`
        - `is_loaded_with_dependencies()`
- `Bundle`
    - a trait representing a set of components
    - a wrapper just for the convenience
- `Commands`
    - represents mutation to be applied to `World`
- `Component`
    - data fields
- `Edges`
    - used as a cache when inserting/removing/taking an entity from an archetype to another.
    - (fields)
        - `insert_bundle: SparseArray<BundleId, ArchetypeAfterBundleInsert>`
        - `remove_bundle: SparseArray<BundleId, Option<ArchetypeId>>`
        - `take_bundle: SparseArray<BundleId, Option<ArchetypeId>>`
- `Entity`
    - just an ID
    - (fields)
        - `index: EntityIndex(NonMaxU32)`
        - `generation: EntityGeneration(u32)`,
- `EntityGeneration(u32)`
    - tracks different versions of an `EntityIndex`
- `EntityLocation`
    - (fields)
        - `archetype_id: ArchetypeId`
        - `archetype_row: ArchetypeRow`
        - `table_id: TableId`
        - `table_row: TableRow`
- `EntityMeta`
    - (fields)
        - `generation: EntityGeneration`
        - `location`
        - `spawned_or_despawned`
- `Entities`
    - (fields)
        - `meta: Vec<EntityMeta>`
- `Event`
- `Handle`
    - not asset itself
    - (variants)
        - `Strong(Arc<StrongHandle>)`
        - Uuid
- `Plugin`
    - trait to specify a module that modifies App
    - (fields)
        - build()
        - ready()
        - finish()
        - cleanup()
        - name()
        - is_unique()
    - e.g.
        - `UiPlugin`
        - `RenderPlugin`
- `PluginState`
    - (variants)
        - `Adding`
        - `Ready`
        - `Finished`
        - `Cleaned`
- `Query`
    - run based on archetypes
- `Resource`
    - a trait for globally unique data
- `Schedule`
- `SparseSet`
- `StorageType`
    - (variants)
        - `Table`
        - `SparseSet`
- `SubApp`
    - (fields)
        - `world: World`
            - hidden from users for parallel execution
        - `plugin_registry: Vec<Box<dyn Plugin>>`
        - `plugin_names: HashSet<String>`
        - `plugin_build_depth`: usize
        - `plugins_state: PluginState`
        - `update_schedule`
        - `extract: Option<ExtractFn>`
- `SubApps`
    - (fields)
        - `main: SubApp`
        - `sub_apps: HasMap<InternedAppLabel, SubApp>`
- `system`
- `Table`
- `World`
    - https://taintedcoders.com/bevy/worlds
    - stores entities, components, resources, and their associated metadata
    - (fields)
        - `id: WorldId(usize)`
        - `entities: Entities`
        - `allocator: EntityAllocator`
        - `components: Components`
        - `component_ids: ComponentIds`
        - `archetypes: Archetypes`
        - `storages: Storages`
        - `bundles: Bundles`
        - `observers: Observers`
        - `command_queue: RawCommandQueue`

### ECS

https://bevyengine.org/learn/quick-start/getting-started/ecs/

```rust
use bevy::prelude::*;

#[derive(Component)]
struct Person;

#[derive(Component)]
struct Name(String);

#[derive(Resource)]
struct GreetTimer(Timer);

pub struct HelloPlugin;


fn add_people(mut commands: Commands) {
    commands.spawn((Person, Name("Elaina Proctor".to_string())));
    commands.spawn((Person, Name("Renzo Hume".to_string())));
    commands.spawn((Person, Name("Zayna Nieves".to_string())));
}

fn update_people(mut query: Query<&mut Name, With<Person>>) {
    for mut name in &mut query {
        if name.0 == "Elaina Proctor" {
            name.0 = "Elaina Hume".to_string();
            break; // We don't need to change any other names.
        }
    }
}

fn greet_people(time: Res<Time>, mut timer: ResMut<GreetTimer>, query: Query<&Name, With<Person>>) {
    // update our timer with the time elapsed since the last update
    // if that caused the timer to finish, we say hello to everyone
    if timer.0.tick(time.delta()).just_finished() {
        for name in &query {
            println!("hello {}!", name.0);
        }
    }
}

impl Plugin for HelloPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GreetTimer(Timer::from_seconds(2.0, TimerMode::Repeating)));
        app.add_systems(Startup, add_people);
        app.add_systems(Update, (update_people, greet_people).chain());
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(HelloPlugin)
        .run();
}

```

- https://taintedcoders.com/bevy/archetypes#starting-without-archetypes

## 3D Rendering

### 3D Scene

https://bevyengine.org/examples/3d-rendering/3d-scene/

## Physics engine

https://github.com/Jondolf/avian

## Virtual geometry / automatic LOD / / meshlet

- https://jms55.github.io/posts/2025-03-27-virtual-geometry-bevy-0-16/
- https://jms55.github.io/posts/2024-11-14-virtual-geometry-bevy-0-15/

## References

- https://bevy.org/learn/contribute/introduction/
- https://taintedcoders.com/
