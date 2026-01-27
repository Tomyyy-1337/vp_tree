use std::{borrow::Borrow, collections::BinaryHeap, vec};

use crate::{Distance, Querry};

/// Vantage-Point Tree (VP-Tree) implementation for efficient nearest neighbor search and radius searches.
/// Requires stored elements to implement the [`Distance`] trait to themselves.
/// Search targets are required to implement [`Distance`] to the stored type.
/// 
/// While constructing the tree takes longer than a naive linear search, nearest neighbor and radius searches are significantly faster 
/// resulting in overall performance gains for multiple searches on the same dataset.
/// 
/// The tree takes 8 bytes of memory per stored element for the distance thresholds, plus the memory required to store the elements themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct VpTree<T> {
    items: Vec<T>,
    nodes: Vec<f64>,
}

impl<T: Distance<T>> VpTree<T> {
    const ROOT: usize = 0;

    /// Constructs a new [`VpTree`] from a [`Vec`] of items. The items are consumed and stored within the tree. 
    /// This constructor uses a single thread. For parallel construction, use [`Self::new_parallel`].
    pub fn new(mut items: Vec<T>) -> Self {
        let mut nodes = vec![0.0; items.len()];
        Self::build_from_points(&mut items, &mut nodes);
        VpTree { items, nodes }
    }   

    /// Constructs a new [`VpTree`] from a [`Vec`] of items using multiple threads. The items are consumed and stored within the tree.
    /// The `threads` parameter specifies the number of threads to use for construction. Powers of 2 are recommended for optimal performance.
    pub fn new_parallel(mut items: Vec<T>, threads: usize) -> Self 
    where
        T: Send,
    {
        let mut nodes = vec![0.0; items.len()];
        Self::build_from_points_par(&mut items, &mut nodes, threads);
        VpTree { items, nodes }
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

        self.search_rec(Self::ROOT, self.items.len(), target, querry.max_items, &mut heap, &mut tau, querry.exclusive);

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
        self.search_nearest_rec(Self::ROOT, self.items.len(), target, &mut best_index, &mut best_distance, false);
        best_index.map(|index| &self.items[index])
    }

    /// Searches for the single nearest neighbor to the target, excluding the target itself if it is present in the tree.
    /// To include the target itself in the results, use [`Self::nearest_neighbor`].
    pub fn nearest_neighbor_exclusive<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best_index = None;
        let mut best_distance = f64::INFINITY;
        self.search_nearest_rec(Self::ROOT, self.items.len(), target, &mut best_index, &mut best_distance, true);
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

    fn build_from_points_par(items: &mut[T], nodes: &mut [f64], threads: usize)
    where 
        T: Send,
    {
        if threads <= 1 {
            return Self::build_from_points(items, nodes);
        }
    
        if items.len() <= 1 {
            return;
        }

        let i = fastrand::usize(..items.len());
        items.swap(0, i);
        let (random_element, slice) = items.split_first_mut().unwrap();
        
        let median = slice.len() / 2;

        let (_, median_item, _) = slice.select_nth_unstable_by(median, |a, b| {
            let dist_a = random_element.distance_heuristic(a);
            let dist_b = random_element.distance_heuristic(b);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let threashold = random_element.distance(median_item);
        nodes[0] = threashold;

        let (left_slice, right_slice) = slice.split_at_mut(median);
        let (left_nodes, right_nodes) = nodes[1..].split_at_mut(median);

        std::thread::scope(|s| {
            s.spawn(|| 
                Self::build_from_points_par(left_slice, left_nodes, threads / 2 + threads % 2)
            );
            Self::build_from_points_par(right_slice, right_nodes, threads / 2);
        });
    }

    fn build_from_points(items: &mut[T], nodes: &mut [f64]) {
        if items.len() <= 1 {
            return;
        }

        let i = fastrand::usize(..items.len());
        items.swap(0, i);
        let (random_element, slice) = items.split_first_mut().unwrap();
        
        let median = slice.len() / 2;

        let (_, median_item, _) = slice.select_nth_unstable_by(median, |a, b| {
            let dist_a = random_element.distance_heuristic(a);
            let dist_b = random_element.distance_heuristic(b);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let threashold = random_element.distance(median_item);
        nodes[0] = threashold;
        
        let (left_slice, right_slice) = slice.split_at_mut(median);
        let (left_nodes, right_nodes) = nodes[1..].split_at_mut(median);

        Self::build_from_points(left_slice, left_nodes);
        Self::build_from_points(right_slice, right_nodes);
    }

    fn search_rec<U: Distance<T>>(
        &self,
        node_index: usize,
        len: usize,
        target: &U,
        k: usize,
        heap: &mut BinaryHeap<HeapItem>,
        tau: &mut f64,
        exclusive: bool
    ) {
        if len == 0 {
            return;
        }

        let threashold = &self.nodes[node_index];
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

        let left = node_index + 1;
        let right = node_index + 1 + (len - 1) / 2;
        let len_left = (len - 1) / 2;
        let right_len = len - 1 - len_left;

        if dist <= *threashold {
            self.search_rec(left, len_left, target, k, heap, tau, exclusive);
            if dist + *tau >= *threashold {
                self.search_rec(right, right_len, target, k, heap, tau, exclusive);
            }
        } else {
            self.search_rec(right, right_len, target, k, heap, tau, exclusive);
            if dist - *tau <= *threashold {
                self.search_rec(left, len_left, target, k, heap, tau, exclusive);
            }
        }
    }

    fn search_nearest_rec<U: Distance<T>>(
        &self,
        node_index: usize,
        len: usize,
        target: &U,
        best_index: &mut Option<usize>,
        best_distance: &mut f64,
        exclusive: bool
    ) {
        if len == 0 {
            return;
        }

        let threashold = &self.nodes[node_index];
        let dist = target.distance(&self.items[node_index]);

        if dist < *best_distance && (!exclusive || dist > 0.0) {
            *best_distance = dist;
            *best_index = Some(node_index);
        }

        let left = node_index + 1;
        let right = node_index + 1 + (len - 1) / 2;
        let len_left = (len - 1) / 2;
        let right_len = len - 1 - len_left;

        if dist <= *threashold {
            self.search_nearest_rec(left, len_left, target, best_index, best_distance, exclusive);
            if dist + *best_distance >= *threashold {
                self.search_nearest_rec(right, right_len, target, best_index, best_distance, exclusive);
            }
        } else {
            self.search_nearest_rec(right, right_len, target, best_index, best_distance, exclusive);
            if dist - *best_distance <= *threashold {
                self.search_nearest_rec(left, len_left, target, best_index, best_distance, exclusive);
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

pub struct HeapItem {
    index: usize,
    distance: f64,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}
