// Goal-oriented action planning.
// Annoying things:
// 1. The way the algorithm is implemented requires the state object to
// implement Hash. That means it isn't possible to use HashMaps as containers
// for the working memory, because HashMaps can't derive Hash. Structures like
// BTreeMap can be used instead.
// 2. Much cloning going on, because one has to keep track of the states that
// have been visited, the minimum cost to reach each state, and the states that
// are to be visited in the frontier. The state object has to be used as a key
// for all the containers keeping track of these things, and must be cloned.
// Preferably, a container that is inexpensive to clone should be used for the
// working memory of the agent who is planning an action.
// 3. The A* implementation I started with treated state transitions as
// retrievable from the final sequence of states the planner makes, because it
// was originally designed for searching in paths of 2D points. Therefore the
// transition to take could be found by simply subtracting two neighbors in the
// path. But when planning fom a set of actions, it would be more difficult, if
// not impossible, to retrieve the set of actions taken just by looking at the
// final states, so the A* trait accounts for both states and the transitions
// between them.
// 4. Someone might want to check only a subset of the properties of the world,
// and not care about the rest. But this implementation hashes a complete copy
// of the state transitioned to. This means that the destination node passed in
// might not exist as a key in the hashmap of seen states. So, when a state that
// matches the goal is found, it is used as the "destination" when backtracking
// along the parents of each node in the path, instead of whatever was passed in
// as the set of goal conditions.

#![feature(test)]
#![feature(associated_consts)]

mod gen;

extern crate rand;
extern crate test;
extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate toml;
#[macro_use] extern crate enum_derive;
#[macro_use] extern crate macro_attr;

use std::f32;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

#[derive(PartialEq)]
struct AStarState<P> {
    cost: f32,
    position: P,
}

impl<P> Eq for AStarState<P>
    where P: PartialEq {}

impl<P> Ord for AStarState<P>
    where P: PartialEq
{
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(self.cost.is_finite());
        assert!(other.cost.is_finite());
        if other.cost > self.cost {
            Ordering::Greater
        } else if other.cost < self.cost {
            Ordering::Less
        } else if other.cost == self.cost {
            Ordering::Equal
        } else {
            unreachable!()
        }
    }
}

impl<P> PartialOrd for AStarState<P>
    where P: PartialEq
{
    fn partial_cmp(&self, other: &AStarState<P>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A generic A* trait that accounts for both states and transitions between
/// them.
pub trait AStar<A, S>
    where S: Hash + Eq + PartialEq + Clone + Debug,
          A: PartialEq + Clone + Debug {

    const CALCULATION_LIMIT: u32 = 200;

    /// The heuristic function used by A*.
    fn heuristic(&self, next_state: &S, goal: &S) -> f32;

    /// The cost from transitioning from a given state using an action.
    fn movement_cost(&self, current_state: &S, action: &A) -> f32;

    /// The set of neighboring states and the transitions to each.
    fn neighbors(&self, current_state: &S) -> Vec<(A, S)>;

    fn goal_reached(&self, current_state: &S, goal: &S) -> bool;
    fn goal_is_reachable(&self, goal: &S) -> bool;

    /// Finds an optimal sequence of transitions, if any, from the start state
    /// to the destination state.
    fn find(&self, from: S, to: &S) -> Vec<A> {
        if from == *to {
            return vec![];
        }

        if !self.goal_is_reachable(&to) {
            return vec![];
        }

        let mut frontier = BinaryHeap::new();
        frontier.push(AStarState { position: from.clone(), cost: 0.0 });
        let mut came_from = HashMap::new();
        let mut cost_so_far = HashMap::new();

        came_from.insert(from.clone(), None);
        cost_so_far.insert(from.clone(), 0.0);

        let mut calculation_steps = 0;
        let mut final_state = None;

        while let Some(current) = frontier.pop() {
            if self.goal_reached(&current.position, &to) {
                final_state = Some(current.position);
                break;
            }
            // Waiting for associated_consts to be stabilized...
            if calculation_steps >= Self::CALCULATION_LIMIT {
                break
            } else {
                calculation_steps += 1;
            }
            let neigh = self.neighbors(&current.position);

            for (action, next_state) in neigh.into_iter() {
                let new_cost = cost_so_far[&current.position] + self.movement_cost(&current.position, &action);
                let val = cost_so_far.entry(next_state.clone()).or_insert(f32::MAX);
                if new_cost < *val {
                    *val = new_cost;
                    let priority = new_cost + self.heuristic(&next_state, &to);
                    frontier.push(AStarState { position: next_state.clone(), cost: priority });
                    came_from.insert(next_state.clone(), Some((action, current.position.clone())));
                }
            }
        }

        // The user might pass in a state with only the properties cared about
        // when checking if the goal is reached. But the data structures work
        // with complete snapshots of states with all variables accounted for.
        // So, we have to use pull out the complete state that was found during
        // the exploration (if any) and use that as the start node for making
        // the returned path.
        match final_state {
            Some(dest) => self.create_result(from, &dest, came_from),
            None       => vec![]
        }
    }

    fn create_result(&self, start: S, goal: &S, came_from: HashMap<S, Option<(A, S)>>) -> Vec<A> {
        let mut current = goal.clone();
        let mut path_buffer = vec![];
        let mut state_history = vec![];
        while current != start {
            match came_from.get(&current) {
                Some(&Some((ref action, ref new_current))) => {
                    current = new_current.clone();
                    path_buffer.push(action.clone());
                    if current != start {
                        state_history.push(new_current.clone());
                    }
                }
                Some(&None) => panic!(
                    "Reached a dead-end before reaching the start node when tracing backwards from the goal to the start."),
                None => {
                    path_buffer = vec![];
                    break
                },
            }
        }

        assert_eq!(None, state_history.iter().find(|p| **p == start),
                   "The path that was found looped back on itself! {:?}", state_history);

        path_buffer.reverse();
        path_buffer
    }
}

/// A trait for objects that can search over the space of possible states and
/// generate an optimal plan for transitioning between start and goal states.
pub trait Planner {
    type Action;
    type State;

    fn get_plan(&self, state: Self::State) -> Vec<Self::Action>;
    fn set_goal(&mut self, goal: Self::State);
}

#[cfg(test)]
mod tests {
    mod toml_util;

    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use std::fmt::{self, Display};
    use self::MyKey::*;
    use self::MyAction::*;

    type PropMap = BTreeMap<MyKey, bool>;

    fn get_state_differences(current: &PropMap, goal: &PropMap) -> u32 {
        assert!(goal.len() <= current.len(), "The goal state specifies more conditions than were given in the starting state!");
        let mut differences = 0;
        for key in goal.keys() {
            if goal.get(key).unwrap() != current.get(key).unwrap() {
                differences += 1;
            }
        }
        differences
    }

    fn transition(current: &MyState, next: &Effects) -> MyState {
        let mut new_state = current.clone();
        for key in next.effects.keys() {
            let val_mut = new_state.props.get_mut(key).unwrap();
            *val_mut = *next.effects.get(key).unwrap();
        }
        new_state
    }

    #[derive(Debug)]
    struct Effects {
        preconditions: PropMap,
        effects: PropMap,
        cost: u32,
    }

    impl Effects {
        pub fn new(cost: u32) -> Self {
            Effects {
                preconditions: PropMap::new(),
                effects: PropMap::new(),
                cost: cost,
            }
        }

        pub fn set_precondition(&mut self, key: MyKey, val: bool) {
            self.preconditions.insert(key, val);
        }

        pub fn set_effect(&mut self, key: MyKey, val: bool) {
            self.effects.insert(key, val);
        }
    }

    struct MyFinder {
        actions: HashMap<MyAction, Effects>,
    }

    impl AStar<MyAction, MyState> for MyFinder {
        fn heuristic(&self, next: &MyState, destination: &MyState) -> f32 {
            get_state_differences(&next.props, &destination.props) as f32
        }

        fn movement_cost(&self, _current: &MyState, action: &MyAction) -> f32 {
            let current_eff = self.actions.get(&action).unwrap();
            current_eff.cost as f32
        }

        fn neighbors(&self, current: &MyState) -> Vec<(MyAction, MyState)> {
            let satisfies_conditions = |effect: &Effects| {
                effect.preconditions.iter().all(|(cond, val)| {
                    current.props.get(cond).unwrap() == val
                })
            };
            let actions = self.actions.iter()
                .filter(|&(_, action_effects)| satisfies_conditions(action_effects))
                .map(|(action, _)| action)
                .cloned().collect::<Vec<MyAction>>();

            let states = actions.iter().cloned()
                .map(|action| transition(&current, self.actions.get(&action).unwrap()))
                .collect::<Vec<MyState>>();

            actions.into_iter().zip(states.into_iter()).collect()
        }

        fn goal_reached(&self, current: &MyState, to: &MyState) -> bool {
            to.props.iter().all(|(cond, val)| current.props.get(cond).unwrap() == val)
        }

        fn goal_is_reachable(&self, _to: &MyState) -> bool {
            true
        }
    }

    macro_attr! {
        #[derive(Clone, Hash, Debug, Eq, PartialEq, Ord, PartialOrd, EnumFromStr!)]
        enum MyKey {
            HasAxe,
            FirewoodOnGround,
            HasFirewood,
            HasMoney,
            InShop,
            InForest,
        }
    }

    #[derive(Clone, Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
    struct MyState {
        props: PropMap,
    }

    impl Display for MyState {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            for (k, v) in self.props.iter() {
                write!(f, "{:?} -> {}\n", k, v)?;
            }
            Ok(())
        }
    }

    macro_attr! {
        #[derive(Hash, Deserialize, Ord, PartialOrd, Eq, PartialEq, Debug, Clone, EnumFromStr!)]
        enum MyAction {
            ChopTree,
            GatherBranches,
            BuyAxe,
            DropAxe,
            GetFirewood,
            SellFirewood,
            GoToForest,
            GoToShop,
        }
    }

    struct MyPlanner {
        goal: Option<MyState>,
        finder: MyFinder,
    }

    impl MyPlanner {
        pub fn new() -> Self {
            let mut actions = HashMap::new();
            let value = toml_util::toml_value_from_file("./data/actions.toml");
            if let toml::Value::Array(ref array) = value["action"] {
                for action in array.iter() {
                    // Parse string as enum
                    let name = action["name"].clone().try_into::<String>().unwrap()
                        .parse::<MyAction>().unwrap();
                    let cost = action["cost"].clone().try_into::<u32>().unwrap();
                    let mut effects = Effects::new(cost);

                    // Read preconditions
                    if let toml::Value::Table(ref pre_table) = action["pre"] {
                        for (pre_name, pre_value) in pre_table.iter() {
                            let key = pre_name.parse::<MyKey>().unwrap();
                            let value = pre_value.clone().try_into::<bool>().unwrap();
                            effects.set_precondition(key, value);
                        }
                    }

                    // Read postconditions
                    if let toml::Value::Table(ref post_table) = action["post"] {
                        for (post_name, post_value) in post_table.iter() {
                            let key = post_name.parse::<MyKey>().unwrap();
                            let value = post_value.clone().try_into::<bool>().unwrap();
                            effects.set_effect(key, value);
                        }
                    }

                    actions.insert(name, effects);
                }
            }

            MyPlanner {
                goal: None,
                finder: MyFinder {
                    actions: actions,
                },
            }
        }
    }

    impl Planner for MyPlanner {
        type Action = MyAction;
        type State = MyState;

        fn get_plan(&self, state: MyState) -> Vec<MyAction> {
            if let Some(ref goal) = self.goal {
                self.finder.find(state, &goal)
            } else {
                vec![]
            }
        }

        fn set_goal(&mut self, goal: MyState) {
            self.goal = Some(goal);
        }
    }

    use test::Bencher;
    use rand::{self, Rng};

    #[test]
    fn test_get_plan() {
        let mut goal_c = BTreeMap::new();

        // Firewood on the ground can only be produced by cutting down a tree.
        goal_c.insert(FirewoodOnGround, true);
        let goal = MyState { props: goal_c };

        // All properties of the starting world must be specified, for now.
        let mut start_c =  BTreeMap::new();
        start_c.insert(HasAxe, false);
        start_c.insert(FirewoodOnGround, false);
        start_c.insert(HasFirewood, false);
        start_c.insert(HasMoney, false);
        start_c.insert(InShop, false);
        start_c.insert(InForest, true);

        let start = MyState { props: start_c };

        let mut planner = MyPlanner::new();
        planner.set_goal(goal);

        let plan = planner.get_plan(start);
        assert_eq!(plan, [GatherBranches, GoToShop, SellFirewood, BuyAxe, GoToForest, ChopTree]);
    }

    #[bench]
    fn bench_get_plan(b: &mut Bencher) {
        let mut rng = rand::thread_rng();

        let mut planner = MyPlanner::new();

        b.iter(|| {
            let mut goal_c = BTreeMap::new();
            goal_c.insert(HasMoney, rng.gen());
            goal_c.insert(HasAxe, rng.gen());
            goal_c.insert(FirewoodOnGround, rng.gen());
            goal_c.insert(HasFirewood, rng.gen());
            goal_c.insert(InShop, true);
            goal_c.insert(InForest, false);
            let goal = MyState { props: goal_c };

            let mut start_c =  BTreeMap::new();
            start_c.insert(HasAxe, rng.gen());
            start_c.insert(FirewoodOnGround, rng.gen());
            start_c.insert(HasFirewood, rng.gen());
            start_c.insert(HasMoney, rng.gen());
            start_c.insert(InShop, true);
            start_c.insert(InForest, false);
            let start = MyState { props: start_c };

            // println!("===Start===\n{}===Goal====\n{}", start, goal);

            planner.set_goal(goal);
            let plan = planner.get_plan(start);
            // println!("Plan: {:?}", plan);
        });
    }
}
