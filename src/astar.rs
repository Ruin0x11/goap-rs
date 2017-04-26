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

const CALCULATION_LIMIT: u32 = 200;

/// A generic A* trait that accounts for both states and transitions between
/// them.
pub trait AStar<A, S>
    where S: Hash + Eq + PartialEq + Clone + Debug,
          A: PartialEq + Clone + Debug {

    /// The heuristic function used by A*.
    fn heuristic(&self, next_state: &S, goal: &S) -> f32;

    /// The cost from transitioning from a given state using an action.
    fn movement_cost(&self, current_state: &S, action: &A) -> f32;

    /// The set of neighboring states and the transitions to each.
    fn neighbors(&self, current_state: &S) -> Vec<(A, S)>;

    fn finished(&self, current_state: &S, goal: &S) -> bool;
    fn goal_is_reachable(&self, goal: &S) -> bool;

    /// Finds an optimal sequence of transitions, if any, from the start state
    /// to the destination state.
    fn find(&self, from: &S, to: &S) -> Vec<A> {
        if from == to {
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
            if self.finished(&current.position, &to) {
                final_state = Some(current.position);
                break;
            }
            // Waiting for associated_consts to be stabilized...
            if calculation_steps >= CALCULATION_LIMIT {
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

        // The user might have passed in a goal state with only the properties
        // cared about when checking if the goal is reached. But the data
        // structures work with complete snapshots of states with all variables
        // accounted for. So, we have to use pull out the complete state that
        // was found during the exploration (if any) and use that as the start
        // node for making the returned path.
        match final_state {
            Some(dest) => self.create_result(from, &dest, came_from),
            None       => vec![]
        }
    }

    fn create_result(&self, start: &S, goal: &S, came_from: HashMap<S, Option<(A, S)>>) -> Vec<A> {
        let mut current = goal.clone();
        let mut path_buffer = vec![];
        let mut state_history = vec![];
        while current != *start {
            match came_from.get(&current) {
                Some(&Some((ref action, ref new_current))) => {
                    current = new_current.clone();
                    path_buffer.push(action.clone());
                    if current != *start {
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

        assert_eq!(None, state_history.iter().find(|p| *p == start),
                   "The path that was found looped back on itself! {:?}", state_history);

        path_buffer.reverse();
        path_buffer
    }
}
