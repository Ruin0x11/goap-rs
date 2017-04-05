#[cfg(test)]
mod tests {
    use ::*;
    use std::collections::BTreeMap;
    use std::fmt::{self, Display};
    use test::Bencher;
    use rand::{self, ThreadRng, Rng};

    type PropMap = BTreeMap<String, bool>;

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

        pub fn set_precondition(&mut self, key: String, val: bool) {
            self.preconditions.insert(key, val);
        }

        pub fn set_effect(&mut self, key: String, val: bool) {
            self.effects.insert(key, val);
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

    struct MyFinder {
        pub actions: HashMap<String, Effects>,
    }

    impl AStar<String, MyState> for MyFinder {
        fn heuristic(&self, next: &MyState, destination: &MyState) -> f32 {
            get_state_differences(&next.props, &destination.props) as f32
        }

        fn movement_cost(&self, _current: &MyState, action: &String) -> f32 {
            let current_eff = self.actions.get(action).unwrap();
            current_eff.cost as f32
        }

        fn neighbors(&self, current: &MyState) -> Vec<(String, MyState)> {
            let satisfies_conditions = |effect: &Effects| {
                effect.preconditions.iter().all(|(cond, val)| {
                    current.props.get(cond).unwrap() == val
                })
            };
            let actions = self.actions.iter()
                .filter(|&(_, action_effects)| satisfies_conditions(action_effects))
                .map(|(action, _)| action)
                .cloned().collect::<Vec<String>>();

            let states = actions.iter().cloned()
                .map(|action| transition(&current, self.actions.get(&action).unwrap()))
                .collect::<Vec<MyState>>();

            actions.into_iter().zip(states.into_iter()).collect()
        }

        fn goal_reached(&self, current: &MyState, to: &MyState) -> bool {
            // TEMP: Deduplicate.
            to.props.iter().all(|(cond, val)| current.props.get(cond).unwrap() == val)
        }

        fn goal_is_reachable(&self, _to: &MyState) -> bool {
            true
        }
    }

    struct MyPlanner {
        goal: Option<MyState>,
        pub keys: Vec<String>,
        pub finder: MyFinder,
    }

    impl MyPlanner {
        pub fn new() -> Self {
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
                let mut effects = Effects::new(cost);

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
                    effects.set_effect(key.clone(), val);
                }
                actions.insert(action_name, effects);
            }
            MyPlanner {
                goal: None,
                keys: keys,
                finder: MyFinder {
                    actions: actions,
                },
            }
        }

    }

    impl Planner for MyPlanner {
        type Action = String;
        type State = MyState;

        fn get_plan(&self, state: MyState) -> Vec<String> {
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

    fn gen_state(rng: &mut ThreadRng, planner: &MyPlanner) -> MyState {
        let mut state_conds = BTreeMap::new();
        for key in planner.keys.iter() {
            state_conds.insert(key.clone(), rng.gen());
        }
        MyState { props: state_conds }
    }

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
