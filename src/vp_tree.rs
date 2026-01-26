use std::{borrow::Borrow, collections::BinaryHeap, vec};

use crate::{Distance, Querry, heap_item::HeapItem};

/// Vantage-Point Tree (VP-Tree) implementation for efficient nearest neighbor search and radius searches.
/// Requires stored elements to implement the [`Distance`] trait to themselves.
/// Search targets are required to implement [`Distance`] to the stored type.
/// 
/// While constructing the tree takes longer than a naive linear search, nearest neighbor and radius searches are significantly faster 
/// resulting in overall performance gains for multiple searches on the same dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct VpTree<T> {
    items: Vec<T>,
    root: OptionalUsize,
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    threashold: f64,
    left: OptionalUsize,
    right: OptionalUsize,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            threashold: 0.0,
            left: OptionalUsize::none(),
            right: OptionalUsize::none(),
        }
    }
}

/// Used to represent an optional usize value without the overhead of `Option<usize>`.
/// The value `usize::MAX` is used to represent `None`. 
#[derive(Debug, Copy, Clone, PartialEq)]
struct OptionalUsize {
    value: usize,
}

impl OptionalUsize {
    fn new_unchecked(value: usize) -> Self {
        OptionalUsize { value }
    }
    
    fn none() -> Self {
        OptionalUsize { value: usize::MAX }
    }

    fn as_option(&self) -> Option<usize> {
        match self.value {
            usize::MAX => None,
            v => Some(v),
        }
    }
}

impl<T: Distance<T>> VpTree<T> {
    /// Constructs a new [`VpTree`] from a [`Vec`] of items. The items are consumed and stored within the tree. 
    /// This constructor uses a single thread. For parallel construction, use [`Self::new_parallel`].
    pub fn new(mut items: Vec<T>) -> Self {
        assert!(items.len() < usize::MAX, "VpTree cannot store more than usize::MAX - 1 items.");
        let mut nodes = vec![Node::default(); items.len()];
        let root = Self::build_from_points(&mut items, 0, &mut nodes);
        VpTree { items, root, nodes }
    }   

    /// Constructs a new [`VpTree`] from a [`Vec`] of items using multiple threads. The items are consumed and stored within the tree.
    pub fn new_parallel(mut items: Vec<T>, threads: usize) -> Self 
    where
        T: Send,
    {
        assert!(items.len() < usize::MAX, "VpTree cannot store more than usize::MAX - 1 items.");
        let mut nodes = vec![Node::default(); items.len()];
        let root = Self::build_from_points_par(&mut items, 0, &mut nodes, threads);
        VpTree { items, root, nodes }
    }

    /// Performs a query on the VpTree using the specified target and query parameters.
    /// Returns a vector of references to the items that match the query criteria.
    pub fn querry<U, Q>(&self, target: &U, querry: Q) -> Vec<&T> 
    where
        U: Distance<T>,
        Q: Borrow<Querry>,
    {
        let querry = querry.borrow();
        let mut heap = BinaryHeap::new();
        let mut tau = querry.max_distance;

        let root = self.root;
        self.search_rec(root, target, querry.max_items, &mut heap, &mut tau, querry.exclusive);

        if querry.sorted {
            heap.into_sorted_vec()
                .into_iter()
                .map(|item| &self.items[item.index])
                .collect()
        } else {
            heap.into_iter()
                .map(|item| &self.items[item.index])
                .collect()
        }
    }

    /// Searches for the single nearest neighbor to the target. Results may include the target itself if it is present in the tree.
    /// To exclude the target itself from the results (distance zero), use [`Self::nearest_neighbor_exclusive`].
    pub fn nearest_neighbor<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best_index = None;
        let mut best_distance = f64::INFINITY;
        self.search_nearest_rec(self.root, target, &mut best_index, &mut best_distance, false);
        best_index.map(|index| &self.items[index])
    }

    /// Searches for the single nearest neighbor to the target, excluding the target itself if it is present in the tree.
    /// To include the target itself in the results, use [`Self::nearest_neighbor`].
    pub fn nearest_neighbor_exclusive<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best_index = None;
        let mut best_distance = f64::INFINITY;
        self.search_nearest_rec(self.root, target, &mut best_index, &mut best_distance, true);
        best_index.map(|index| &self.items[index])
    }

    /// Returns a reference to all items stored in the VpTree. The items are stored in an arbitrary order.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Consumes the [`VpTree`] and returns the items stored within it. The items are returned in an arbitrary order.
    pub fn into_items(self) -> Vec<T> {
        self.items
    }

    fn build_from_points_par(items: &mut[T], offset: usize, nodes: &mut [Node], threads: usize) -> OptionalUsize
    where 
        T: Send,
    {
        if threads <= 1 {
            return Self::build_from_points(items, offset, nodes);
        }
        
        let num_items = items.len();    

        if num_items == 0 {
            return OptionalUsize::none();
        }

        if num_items == 1 {
            return OptionalUsize::new_unchecked(offset)
        }

        let i = fastrand::usize(..num_items);
        items.swap(0, i);
        let (random_element, slice) = items.split_first_mut().unwrap();
        
        let median = num_items / 2 - 1;

        let (_, median_item, _) = slice.select_nth_unstable_by(median, |a, b| {
            let dist_a = random_element.distance_heuristic(a);
            let dist_b = random_element.distance_heuristic(b);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let threashold = random_element.distance(median_item);
        let (left_slice, right_slice) = slice.split_at_mut(median);
        let (first_node, rest_nodes) = nodes.split_first_mut().unwrap();
        let (left_nodes, right_nodes) = rest_nodes.split_at_mut(median);

        first_node.threashold = threashold;
        let right_offset = offset + left_slice.len() + 1;
        let (left_index, right_index) = std::thread::scope(|s| {
            let left_handle = s.spawn(|| {
                Self::build_from_points_par(left_slice, offset + 1, left_nodes, threads / 2 + threads % 2)
            });
            let right_index = Self::build_from_points_par(right_slice, right_offset, right_nodes, threads / 2);
            (left_handle.join().unwrap(), right_index)
        });
        first_node.left = left_index;
        first_node.right = right_index;
        OptionalUsize::new_unchecked(offset)
    }

    fn build_from_points(items: &mut[T], offset: usize, nodes: &mut [Node]) -> OptionalUsize {
        let num_items = items.len();    

        if num_items == 0 {
            return OptionalUsize::none();
        }

        if num_items == 1 {
            return OptionalUsize::new_unchecked(offset)
        }

        let i = fastrand::usize(..num_items);
        items.swap(0, i);
        let (random_element, slice) = items.split_first_mut().unwrap();
        
        let median = num_items / 2 - 1;

        let (_, median_item, _) = slice.select_nth_unstable_by(median, |a, b| {
            let dist_a = random_element.distance_heuristic(a);
            let dist_b = random_element.distance_heuristic(b);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let threashold = random_element.distance(median_item);
        let (left_slice, right_slice) = slice.split_at_mut(median);
        let (first_node, rest_nodes) = nodes.split_first_mut().unwrap();
        let (left_nodes, right_nodes) = rest_nodes.split_at_mut(median);

        first_node.threashold = threashold;
        let left_index = Self::build_from_points(left_slice, offset + 1, left_nodes);
        let right_index = Self::build_from_points(right_slice, offset + left_slice.len() + 1, right_nodes);
        first_node.left = left_index;
        first_node.right = right_index;
        OptionalUsize::new_unchecked(offset)
    }

    fn search_rec<U: Distance<T>>(
        &self,
        node: OptionalUsize,
        target: &U,
        k: usize,
        heap: &mut BinaryHeap<HeapItem>,
        tau: &mut f64,
        exclusive: bool
    ) {
        if let Some(node_index) = node.as_option() {
            let Node { threashold, left, right } = &self.nodes[node_index];
            let dist = target.distance(&self.items[node_index]);

            if dist <= *tau && (!exclusive || dist > 0.0) {
                if heap.len() == k {
                    heap.pop();
                }
                heap.push(HeapItem { index: node_index, distance: dist });
                if heap.len() == k {
                    *tau = heap.peek().unwrap().distance;
                }
            }

            if dist <= *threashold {
                self.search_rec(*left, target, k, heap, tau, exclusive);
                if dist + *tau >= *threashold {
                    self.search_rec(*right, target, k, heap, tau, exclusive);
                }
            } else {
                self.search_rec(*right, target, k, heap, tau, exclusive);
                if dist - *tau <= *threashold {
                    self.search_rec(*left, target, k, heap, tau, exclusive);
                }
            }
        }
    }

    fn search_nearest_rec<U: Distance<T>>(
        &self,
        node: OptionalUsize,
        target: &U,
        best_index: &mut Option<usize>,
        best_distance: &mut f64,
        exclusive: bool
    ) {
        if let Some(node_index) = node.as_option() {
            let Node { threashold, left, right } = &self.nodes[node_index];
            let dist = target.distance(&self.items[node_index]);

            if dist < *best_distance && (!exclusive || dist > 0.0) {
                *best_distance = dist;
                *best_index = Some(node_index);
            }

            if dist <= *threashold {
                self.search_nearest_rec(*left, target, best_index, best_distance, exclusive);
                if dist + *best_distance >= *threashold {
                    self.search_nearest_rec(*right, target, best_index, best_distance, exclusive);
                }
            } else {
                self.search_nearest_rec(*right, target, best_index, best_distance, exclusive);
                if dist - *best_distance <= *threashold {
                    self.search_nearest_rec(*left, target, best_index, best_distance, exclusive);
                }
            }
        }
    }
}

impl<T: Distance<T>> FromIterator<T> for VpTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        VpTree::new(items)
    }
}