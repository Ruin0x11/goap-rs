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

mod astar;

extern crate serde;
#[macro_use] extern crate serde_derive;

#[cfg(test)]
#[macro_use] extern crate enum_derive;
#[cfg(test)]
#[macro_use] extern crate macro_attr;
#[cfg(test)]
extern crate rand;
#[cfg(test)]
extern crate toml;

use std::f32;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use astar::AStar;

type PropMap<K, V> = BTreeMap<K, V>;

pub trait Effect {
    fn cost(&self) -> u32;
}

/// A trait for objects that can search over the space of possible states and
/// generate an optimal plan for transitioning between start and goal states.
pub trait Planner<K, V, A, S, E>
    where E: Effect {
    fn get_plan(&self, state: S) -> Vec<A>;
    fn set_goal(&mut self, goal: S);

    fn get_actions(&self) -> Vec<&A>;
    fn actions(&self, action: &A) -> &E;
    fn is_neighbor(&self, current: &S, action: &A) -> bool;
    fn closeness_to(&self, current: &S, to: &S) -> u32;
    fn transition(&self, current: &S, next: &A) -> S;
    fn plan_found(&self, current: &S, goal: &S) -> bool;
}

impl<K, V, A> astar::AStar<A, GoapState<K, V>> for GoapPlanner<K, V, A>
    where K: Ord + PartialOrd + Hash + Eq + PartialEq + Clone + Debug,
          V: Hash + Eq + PartialEq + Clone + Debug,
          A: Hash + Eq + PartialEq + Clone + Debug {
    fn heuristic(&self, next: &GoapState<K, V>, destination: &GoapState<K, V>) -> f32 {
        self.closeness_to(&next, &destination) as f32
    }

    fn movement_cost(&self, _current: &GoapState<K, V>, action: &A) -> f32 {
        let current_eff = self.actions(&action);
        current_eff.cost() as f32
    }

    fn neighbors(&self, current: &GoapState<K, V>) -> Vec<(A, GoapState<K, V>)> {
        let valid_actions = self.get_actions().iter()
            .filter(|&action| self.is_neighbor(current, action))
            .map(|&a| a )
            .cloned().collect::<Vec<A>>();

        let states = valid_actions.iter().cloned()
            .map(|action| self.transition(&current, &action))
            .collect::<Vec<GoapState<K, V>>>();

        valid_actions.into_iter().zip(states.into_iter()).collect()
    }

    fn goal_reached(&self, from: &GoapState<K, V>, to: &GoapState<K, V>) -> bool {
        self.plan_found(from, to)
    }

    fn goal_is_reachable(&self, _to: &GoapState<K, V>) -> bool {
        true
    }
}

#[derive(Serialize, Deserialize, Clone, Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub struct GoapState<K: Ord, V> {
    pub props: PropMap<K, V>,
}

#[derive(Debug)]
pub struct GoapEffects<K, V> {
    preconditions: PropMap<K, V>,
    postconditions: PropMap<K, V>,
    cost: u32,
}

impl<K: Ord + PartialOrd, V> GoapEffects<K, V> {
    pub fn new(cost: u32) -> Self {
        GoapEffects {
            preconditions: PropMap::new(),
            postconditions: PropMap::new(),
            cost: cost,
        }
    }

    pub fn set_precondition(&mut self, key: K, val: V) {
        self.preconditions.insert(key, val);
    }

    pub fn set_postcondition(&mut self, key: K, val: V) {
        self.postconditions.insert(key, val);
    }
}


impl<K, V> Effect for GoapEffects<K, V> {
    fn cost(&self) -> u32 {
        self.cost
    }
}

pub struct GoapPlanner<K: Ord, V, A> {
    pub goal: Option<GoapState<K, V>>,
    pub actions: HashMap<A, GoapEffects<K, V>>,
}

impl<K, V, A> Planner<K, V, A, GoapState<K, V>, GoapEffects<K, V>> for
    GoapPlanner<K, V, A>
    where K: Ord + PartialOrd + Hash + Eq + PartialEq + Clone + Debug,
          V: Hash + Eq + PartialEq + Clone + Debug,
          A: Hash + Eq + PartialEq + Clone + Debug {
    fn get_plan(&self, state: GoapState<K, V>) -> Vec<A> {
        if let Some(ref goal) = self.goal {
            self.find(state, &goal)
        } else {
            vec![]
        }
    }

    fn set_goal(&mut self, goal: GoapState<K, V>) {
        self.goal = Some(goal);
    }

    fn get_actions(&self) -> Vec<&A> {
        self.actions.keys().collect()
    }

    fn actions(&self, action: &A) -> &GoapEffects<K, V> {
        self.actions.get(action).unwrap()
    }

    fn closeness_to(&self, current: &GoapState<K, V>, goal: &GoapState<K, V>) -> u32 {
        assert!(goal.props.len() <= current.props.len(), "The goal state specifies more conditions than were given in the starting state!");
        let mut differences = 0;
        for key in goal.props.keys() {
            if goal.props.get(key).unwrap() != current.props.get(key).unwrap() {
                differences += 1;
            }
        }
        differences
    }

    fn transition(&self, current: &GoapState<K, V>, next: &A) -> GoapState<K, V> {
        let effects = self.actions.get(next).unwrap();
        let mut new_state = current.clone();
        for key in effects.postconditions.keys() {
            let effect_val = effects.postconditions.get(key).unwrap().clone();
            if !new_state.props.contains_key(key) {
                new_state.props.insert(key.clone(), effect_val);
            } else {
                let val_mut = new_state.props.get_mut(key).unwrap();
                *val_mut = effect_val;
            }
        }
        new_state
    }

    fn is_neighbor(&self, current: &GoapState<K, V>, action: &A) -> bool {
        let effects = self.actions.get(action).unwrap();
        effects.preconditions.iter().all(|(cond, val)| {
            current.props.get(cond).map_or(false, |r| r == val)
        })
    }

    fn plan_found(&self, current: &GoapState<K, V>, to: &GoapState<K, V>) -> bool {
        to.props.iter().all(|(cond, val)| current.props.get(cond).map_or(false, |r| r == val))
    }

}

#[cfg(test)]
mod tests {
    mod toml_util;

    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use self::MyKey::*;
    use self::MyAction::*;
    use rand::{self, ThreadRng, Rng};

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

    fn planner_from_toml() -> GoapPlanner<MyKey, bool, MyAction> {
        let mut actions = HashMap::new();
        let value = toml_util::toml_value_from_file("./data/actions.toml");
        if let toml::Value::Array(ref array) = value["action"] {
            for action in array.iter() {
                // Parse string as enum
                let name = action["name"].clone().try_into::<String>().unwrap()
                    .parse::<MyAction>().unwrap();
                let cost = action["cost"].clone().try_into::<u32>().unwrap();
                let mut effects = GoapEffects::new(cost);

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
                        effects.set_postcondition(key, value);
                    }
                }

                actions.insert(name, effects);
            }
        }

        GoapPlanner {
            goal: None,
            actions: actions,
        }
    }

    #[test]
    fn test_get_plan() {
        let mut goal_c = BTreeMap::new();

        // Firewood on the ground can only be produced by cutting down a tree.
        goal_c.insert(FirewoodOnGround, true);
        let goal = GoapState { props: goal_c };

        // All properties of the starting world must be specified, for now.
        let mut start_c =  BTreeMap::new();
        start_c.insert(HasAxe, false);
        start_c.insert(FirewoodOnGround, false);
        start_c.insert(HasFirewood, false);
        start_c.insert(HasMoney, false);
        start_c.insert(InShop, false);
        start_c.insert(InForest, true);

        let start = GoapState { props: start_c };

        let mut planner = planner_from_toml();
        planner.set_goal(goal);

        let plan = planner.get_plan(start);
        assert_eq!(plan, [GatherBranches, GoToShop, SellFirewood, BuyAxe, GoToForest, ChopTree]);
    }

    fn get_random_planner() -> GoapPlanner<String, bool, String> {
        let mut actions = HashMap::new();
        let mut rng = rand::thread_rng();
        let rand_str = || rand::thread_rng().gen_ascii_chars().take(20).collect::<String>();

        let mut keys = Vec::new();

        for _ in 0..10 {
            keys.push(rand_str());
        }

        for _ in 0..100 {
            let action_name = rand_str();
            let cost = rng.gen_range(1, 10);
            let mut effects = GoapEffects::new(cost);

            // Preconditions
            for _ in 0..rng.gen_range(1, 5) {
                let key = rng.choose(&keys).unwrap();
                let val = rng.gen();
                effects.set_precondition(key.clone(), val);
            }

            // Postconditions
            for _ in 0..rng.gen_range(1, 20) {
                let key = rng.choose(&keys).unwrap();
                let val = rng.gen();
                effects.set_postcondition(key.clone(), val);
            }
            actions.insert(action_name, effects);
        }
        GoapPlanner {
            goal: None,
            actions: actions,
        }
    }

    fn gen_state(rng: &mut ThreadRng, planner: &GoapPlanner<String, bool, String>) -> GoapState<String, bool> {
        let mut state_conds = BTreeMap::new();
        for key in planner.actions.keys() {
            state_conds.insert(key.clone(), rng.gen());
        }
        GoapState { props: state_conds }
    }

    #[cfg(never)]
    #[test]
    fn test_generated() {
        let mut planner = get_random_planner();
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let goal = gen_state(&mut rng, &planner);
            let start = gen_state(&mut rng, &planner);
            planner.set_goal(goal);
            let plan = planner.get_plan(start);
            println!("Plan: {:?}", plan);
        }
    }

    #[cfg(never)]
    #[bench]
    fn benchmark_generated(b: &mut Bencher) {
        let mut planner = MyPlanner::new();
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let goal = gen_state(&mut rng, &planner);
            let start = gen_state(&mut rng, &planner);
            planner.set_goal(goal);
            let plan = planner.get_plan(start);
            // println!("Plan: {:?}", plan);
        });
    }
}
