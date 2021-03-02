use crate::graph::NodeIndex;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt::Debug;
use std::iter;
use std::ops::{Add, Index, IndexMut};

/// Result with boxed error as trait object.
pub type GenericResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[cfg(test)]
pub(crate) type TestResult = GenericResult<()>;

/// Numerical type that can either be an unsigned number or positive infinity.
// infinity is encoded as `-1`; all other negative values are illegal
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NaturalOrInfinite(i32);

impl PartialOrd for NaturalOrInfinite {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NaturalOrInfinite {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 < 0 && other.0 < 0 {
            Ordering::Equal
        } else if other.0 < 0 {
            Ordering::Less
        } else if self.0 < 0 {
            Ordering::Greater
        } else {
            self.0.cmp(&other.0)
        }
    }
}

impl Add for NaturalOrInfinite {
    type Output = NaturalOrInfinite;

    fn add(self, rhs: Self) -> Self::Output {
        if self.0 < 0 || rhs.0 < 0 {
            NaturalOrInfinite(-1)
        } else {
            NaturalOrInfinite(self.0 + rhs.0)
        }
    }
}

impl NaturalOrInfinite {
    pub fn infinity() -> Self {
        NaturalOrInfinite(-1)
    }
}

impl From<u32> for NaturalOrInfinite {
    fn from(val: u32) -> Self {
        NaturalOrInfinite(val as i32)
    }
}

impl Default for NaturalOrInfinite {
    fn default() -> Self {
        NaturalOrInfinite::infinity()
    }
}

impl NaturalOrInfinite {
    /// Returns the value as a `u32` if it is finite.
    /// Panics in case `self == NaturalOrInfinite::infinity()`.
    pub fn finite_value(&self) -> u32 {
        if self.0 < 0 {
            panic!("infinite value");
        }
        self.0 as u32
    }
}

pub struct Combinations<'a, T: Copy> {
    indices: Vec<usize>,
    elements: &'a [T],
    count_position: usize,
    buffer: Vec<T>,
    first_iteration: bool,
}

/// Return all combinations (subsets) of the given length.
/// Inspired by [this](https://github.com/rust-itertools/itertools/blob/master/src/combinations.rs).
pub fn combinations<T: Copy>(elements: &[T], n: usize) -> Combinations<T> {
    assert!(n <= elements.len(), "n cannot be larger than #elements");
    Combinations {
        elements,
        indices: (0..n).collect(),
        count_position: if n > 0 { n - 1 } else { 0 },
        buffer: Vec::with_capacity(n),
        first_iteration: true,
    }
}

impl<'a, T: Copy> Combinations<'a, T> {
    /// Check if the index at the given index can be incremented any further.
    fn is_index_done(&self, index: usize) -> bool {
        self.indices[index] >= self.elements.len() - (self.indices.len() - index)
    }

    /// Update buffer element at the position `index`.
    fn update_buffer(&mut self, index: usize) {
        self.buffer[index] = self.elements[self.indices[index]];
    }

    /// Go to the next subset.
    fn increment(&mut self) -> bool {
        if !self.is_index_done(self.count_position) {
            self.indices[self.count_position] += 1;
            self.update_buffer(self.count_position);
        } else {
            let mut i = self.count_position;
            while self.is_index_done(i) {
                if i == 0 {
                    return false;
                }
                i -= 1;
            }
            self.count_position = i;
            self.indices[self.count_position] += 1;
            self.update_buffer(self.count_position);
            for j in self.count_position + 1..self.indices.len() {
                self.indices[j] = self.indices[j - 1] + 1;
                self.update_buffer(j);
            }
            self.count_position = self.indices.len() - 1;
        }
        true
    }

    // This can't be the "real" iterator implementation because the iterator protocol is too
    // restrictive to allow returning references to the iterator itself so I would have to allocate
    // each time which the caller might not want to.
    // Therefore I made this function for the cases where it is not necessary to own the returned
    // subset.
    pub(crate) fn next_buf(&mut self) -> Option<&[T]> {
        if self.first_iteration {
            for &i in &self.indices {
                self.buffer.push(self.elements[i]);
            }
            self.first_iteration = false;
            Some(&self.buffer)
        } else if self.count_position < self.indices.len() && self.increment() {
            Some(&self.buffer)
        } else {
            None
        }
    }
}

impl<'a, T: Copy> Iterator for Combinations<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_buf().map(<[_]>::to_vec)
    }
}

pub struct NonTrivialSubsets<'a, T: Copy> {
    element_count: usize,
    combinations: Combinations<'a, T>,
    elements: &'a [T],
}

impl<'a, T: Copy> NonTrivialSubsets<'a, T> {
    pub fn new(elements: &'a [T]) -> Self {
        Self {
            element_count: 1,
            combinations: combinations(elements, 1),
            elements,
        }
    }
}

pub fn non_trivial_subsets<T: Copy>(elements: &[T]) -> NonTrivialSubsets<T> {
    NonTrivialSubsets::new(elements)
}

impl<'a, T: Copy> Iterator for NonTrivialSubsets<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        match self.combinations.next_buf() {
            None => {
                if self.element_count < self.elements.len() - 1 {
                    self.element_count += 1;
                    self.combinations = combinations(self.elements, self.element_count);
                    self.combinations.next_buf().map(<[_]>::to_vec)
                } else {
                    None
                }
            }
            Some(res) => Some(res.to_vec()),
        }
    }
}

pub struct Vector<T: Default + Clone> {
    vec: Vec<T>,
    dimensions: Vec<usize>,
    products: Vec<usize>, // for each dimension the product of the following dimensions
}

impl<T: Default + Clone> Vector<T> {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let size = dimensions.iter().product();
        let mut vec = Vec::new();
        // This vector is impossible to grow through the current API so it shouldn't reserve more
        // space for future insertions.
        vec.reserve_exact(size);
        for _ in 0..size {
            vec.push(T::default());
        }
        vec.shrink_to_fit();
        let products = (0..dimensions.len())
            .map(|d| dimensions[d + 1..].iter().product())
            .collect();
        Vector {
            dimensions,
            vec,
            products,
        }
    }

    fn raw_index(&self, index: &[usize]) -> usize {
        debug_assert!(
            index
                .iter()
                .enumerate()
                .all(|(ii, &i)| i < self.dimensions[ii]),
            format!("index out of range: {:?}", index)
        );
        index
            .iter()
            .copied()
            .enumerate()
            .map(|(ii, i)| self.products[ii] * i)
            .sum()
    }
}

impl<T: Default + Clone> Index<&[usize]> for Vector<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let raw_index = self.raw_index(index);
        &self.vec[raw_index]
    }
}

impl<T: Default + Clone> IndexMut<&[usize]> for Vector<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let raw_index = self.raw_index(index);
        &mut self.vec[raw_index]
    }
}

/// A map that maps subsets of a certain size of a set of indices (i.e. `usize`s from 0..n).
/// E.g. with a subset size of `2` and the set of indices `{0, 1, 2, 3}` you could do something like
/// this `indexSetMap[&mut [1, 2]] = indexSetMap[&mut [0, 1]]`.
/// Note that indexing mutates the index. The reason that the index is not moved is so that the
/// caller can reuse the allocation. Indexing by `Vec`s instead of slices is also supported; in that
/// case the index value is moved.
pub struct IndexSetMap<T: Copy + Default> {
    map: Vector<T>,
    subset_size: usize,
}

impl<T: Copy + Default> IndexSetMap<T> {
    pub fn new(num_elements: usize, subset_size: usize) -> Self {
        // Since we're talking about sets the dimensions get smaller towards the end.
        // Indices are always going to be ordered, so a two-dimensional IndexSetMap over the
        // indices {0, 1, 2, 3, 4} can never have 0 as its second index.
        // But there also has to be enough elements "left" afterwards so it could also not have
        // 4 as the first index because then there wouldn't be a bigger element for the second
        // dimension.
        // Therefore every dimension can at most have `#elements - (subset_size - 1)`
        let dimensions = iter::repeat(num_elements - (subset_size - 1))
            .take(subset_size)
            .collect();
        Self {
            map: Vector::new(dimensions),
            subset_size,
        }
    }

    pub fn subset_size(&self) -> usize {
        self.subset_size
    }

    fn calculate_index(index: &mut [usize]) {
        #[cfg(debug_assertions)]
        {
            let mut index2 = index.to_vec();
            index2.dedup();
            debug_assert!(index2.len() == index.len())
        }
        index.sort_unstable();
        index.iter_mut().enumerate().for_each(|(ii, i)| *i -= ii);
    }
}

impl<T: Copy + Default> Index<Vec<usize>> for IndexSetMap<T> {
    type Output = T;

    fn index(&self, mut index: Vec<usize>) -> &Self::Output {
        Self::calculate_index(&mut index);
        &self.map[&index]
    }
}

impl<T: Copy + Default> IndexMut<Vec<usize>> for IndexSetMap<T> {
    fn index_mut(&mut self, mut index: Vec<usize>) -> &mut Self::Output {
        Self::calculate_index(&mut index);
        &mut self.map[&index]
    }
}

/// List of [IndexSetMaps] that can be accessed with an offset.
/// Subsets below the offset could e.g. be those that are considered "trivial" and thus need not
/// be considered. Cf. the usage in [crate::steiner_tree::dreyfus_wagner].
pub(crate) struct IndexSetMaps<T: Copy + Default> {
    maps: Vec<IndexSetMap<T>>,
    offset: usize,
}

impl<T: Copy + Default> IndexSetMaps<T> {
    pub fn new(offset: usize) -> Self {
        Self {
            maps: vec![],
            offset,
        }
    }

    pub fn push(&mut self, map: IndexSetMap<T>) {
        assert_eq!(self.maps.len() + self.offset, map.subset_size());
        self.maps.push(map);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.maps.len()
    }
}

impl<T: Copy + Default> Index<Vec<usize>> for IndexSetMaps<T> {
    type Output = T;

    fn index(&self, index: Vec<usize>) -> &Self::Output {
        &self.maps[index.len() - self.offset][index]
    }
}

impl<T: Copy + Default> IndexMut<Vec<usize>> for IndexSetMaps<T> {
    fn index_mut(&mut self, index: Vec<usize>) -> &mut Self::Output {
        &mut self.maps[index.len() - self.offset][index]
    }
}

pub(crate) fn sorted<T: Ord>(elements: &[T]) -> bool {
    for (a, b) in elements.iter().zip(elements.iter().skip(1)) {
        if a > b {
            return false;
        }
    }
    true
}

#[derive(Debug, PartialEq, Eq)]
pub struct PriorityValuePair<P: Ord, V: Ord> {
    pub value: V,
    pub priority: P,
}

impl<P: Ord, V: Ord> Ord for PriorityValuePair<P, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        // compare node as well for consistency with PartialEq/Eq
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.value.cmp(&other.value))
    }
}

impl<P: Ord, V: Ord> PartialOrd for PriorityValuePair<P, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns an edge `(a, b)` such that `a < b`.
pub fn edge(from: NodeIndex, to: NodeIndex) -> (NodeIndex, NodeIndex) {
    assert_ne!(from, to);
    (from.min(to), from.max(to))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::graph::tests::steiner_example_wiki;
    use crate::shortest_paths::ShortestPathMatrix;

    #[test]
    fn test_natural_or_infinite() {
        let inf = NaturalOrInfinite::infinity();
        assert!(NaturalOrInfinite::from(100_000_000) < inf);
        assert_eq!(inf, inf);
        let one = NaturalOrInfinite::from(1);
        let forty_one = NaturalOrInfinite::from(41);
        let forty_two = NaturalOrInfinite::from(42);
        assert_eq!(one + forty_one, forty_two);
        assert_eq!(forty_two + inf, inf);
        assert_eq!(inf + forty_two, inf);
    }

    #[test]
    fn test_combinations_trivial() {
        let empty: &[()] = &[];
        assert_eq!(combinations(empty, 0).collect::<Vec<_>>(), vec![[]]);
        let one = &[1];
        assert_eq!(combinations(one, 0).collect::<Vec<_>>(), vec![[]]);
        assert_eq!(combinations(one, 1).collect::<Vec<_>>(), vec![[1]]);
        let two = &[1, 2];
        assert_eq!(combinations(two, 0).collect::<Vec<_>>(), vec![[]]);
        assert_eq!(combinations(two, 1).collect::<Vec<_>>(), vec![[1], [2]]);
        assert_eq!(combinations(two, 2).collect::<Vec<_>>(), vec![[1, 2]]);
    }

    #[test]
    fn test_combinations() {
        let four = &[1, 2, 3, 4];
        assert_eq!(combinations(four, 0).collect::<Vec<_>>(), vec![[]]);
        assert_eq!(
            combinations(four, 1).collect::<Vec<_>>(),
            vec![[1], [2], [3], [4]]
        );
        assert_eq!(
            combinations(four, 2).collect::<Vec<_>>(),
            vec![[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        );
        assert_eq!(
            combinations(four, 3).collect::<Vec<_>>(),
            vec![[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        );
        assert_eq!(
            combinations(four, 4).collect::<Vec<_>>(),
            vec![[1, 2, 3, 4]]
        );
    }

    #[test]
    fn test_nontrivial_subsets() {
        let four = &[1, 2, 3, 4];
        assert_eq!(
            NonTrivialSubsets::new(four).collect::<Vec<_>>(),
            vec![
                vec![1],
                vec![2],
                vec![3],
                vec![4],
                vec![1, 2],
                vec![1, 3],
                vec![1, 4],
                vec![2, 3],
                vec![2, 4],
                vec![3, 4],
                vec![1, 2, 3],
                vec![1, 2, 4],
                vec![1, 3, 4],
                vec![2, 3, 4]
            ]
        );
    }

    #[test]
    fn test_combinations_sorted() {
        let lots: Vec<u32> = (1..=1000).collect();
        for n in 1..1000 {
            for (_, comb) in (0..3).zip(combinations(&lots, n)) {
                assert!(sorted(&comb))
            }
        }
    }

    #[test]
    fn test_raw_index() {
        let vec: Vector<String> = Vector::new(vec![4, 3, 2]);
        assert_eq!(vec.raw_index(&[1, 2, 1]), 11);
        assert_eq!(vec.raw_index(&[2, 1, 0]), 14);
        assert_eq!(vec.raw_index(&[3, 2, 1]), 23);
        assert_eq!(vec.raw_index(&[1, 1, 1]), 9);
        let vec: Vector<String> = Vector::new(vec![42]);
        assert_eq!(vec.raw_index(&[21]), 21);
        let vec: Vector<&str> = Vector::new(vec![12, 24, 7, 17, 13, 5]);
        let index = &[9, 21, 5, 12, 7, 3];
        assert_eq!(
            vec.raw_index(index),
            9 * (24 * 7 * 17 * 13 * 5)
                + 21 * (7 * 17 * 13 * 5)
                + 5 * (17 * 13 * 5)
                + 12 * (13 * 5)
                + 7 * 5
                + 3
        );
    }

    #[test]
    fn test_vector() {
        let mut vec: Vector<&str> = Vector::new(vec![12, 24, 7, 17, 13, 5]);
        let index = &[9, 21, 5, 12, 7, 3];
        vec[index] = "parrot";
        assert_eq!(vec[index], "parrot");
        let index = &[9, 0, 5, 0, 7, 3];
        vec[index] = "doggo";
        assert_eq!(vec[index], "doggo");
        let mut vec: Vector<usize> = Vector::new(vec![10]);
        for i in 0..10 {
            vec[&[i]] = i;
        }
        for i in 0..10 {
            assert_eq!(vec[&[i]], i);
        }
    }

    #[test]
    #[should_panic]
    fn test_index_set_map_wrong_length_1() {
        let map = IndexSetMap::new(2, 1);
        map[vec![0, 1]]
    }

    #[test]
    #[should_panic]
    fn test_index_set_map_wrong_length_2() {
        let map = IndexSetMap::new(100, 3);
        map[vec![0, 1, 2, 3]]
    }

    #[test]
    #[should_panic]
    fn test_index_set_map_out_of_bounds_1() {
        let map = IndexSetMap::new(0, 1);
        map[vec![0]]
    }

    #[test]
    #[should_panic]
    fn test_index_set_map_out_of_bounds_2() {
        let map = IndexSetMap::new(1, 1);
        map[vec![1]]
    }

    #[test]
    #[should_panic]
    fn test_index_set_map_out_of_bounds_3() {
        let map = IndexSetMap::new(100, 3);
        map[vec![0, 100, 0]]
    }

    #[test]
    #[should_panic]
    fn test_index_set_map_not_a_set() {
        let map = IndexSetMap::new(100, 3);
        map[vec![1, 3, 3]]
    }

    #[test]
    fn test_index_set_map_1() {
        let mut map = IndexSetMap::new(100, 3);
        map[vec![1, 2, 3]] = "frog";
        let index = vec![3, 1, 2];
        assert_eq!(map[index], "frog");
        map[vec![42usize, 21usize, 10usize]] = "dragonfly";
        assert_eq!(map[vec![10usize, 21usize, 42usize]], "dragonfly");
    }

    #[test]
    fn test_index_set_map_2() {
        let mut map = IndexSetMap::new(10, 3);
        let ten: Vec<_> = (0..9).collect();
        for subset in combinations(&ten, 3) {
            let val = subset[0] * 100 + subset[1] * 10 + subset[0];
            map[subset] = val;
        }
        for subset in combinations(&ten, 3) {
            let val = subset[0] * 100 + subset[1] * 10 + subset[0];
            assert_eq!(map[subset], val);
        }
    }

    #[test]
    fn test_index_set_map_default() {
        let mut map: IndexSetMap<NaturalOrInfinite> = IndexSetMap::new(12, 2);
        assert_eq!(map[vec![0, 11]], NaturalOrInfinite::default());
        map[vec![11, 0]] = NaturalOrInfinite(42);
        assert_eq!(map[vec![0, 11]], NaturalOrInfinite(42));
    }

    #[test]
    fn test_index_set_maps() {
        let mut maps: IndexSetMaps<usize> = IndexSetMaps::new(1);
        let elements: Vec<_> = (0..10).collect();
        for subset in non_trivial_subsets(&elements) {
            if maps.len() < subset.len() {
                maps.push(IndexSetMap::new(10, subset.len()));
            }
            let val = subset.iter().sum::<usize>() - subset[0] & 0xbadf00d_usize;
            maps[subset] = val;
        }
        for subset in non_trivial_subsets(&elements) {
            let val = subset.iter().sum::<usize>() - subset[0] & 0xbadf00d;
            assert_eq!(maps[subset], val);
        }
    }

    #[test]
    fn test_index_set_maps_shortest_paths() -> TestResult {
        let graph = steiner_example_wiki()?;
        let shortest_paths = ShortestPathMatrix::new(&graph);
        let mut maps: IndexSetMaps<NaturalOrInfinite> = IndexSetMaps::new(2);
        maps.push(IndexSetMap::new(graph.num_nodes(), 2));
        let nodes = graph.node_indices().collect::<Vec<_>>();
        for pair in combinations(&nodes, 2) {
            let distance = shortest_paths[pair[0]][pair[1]].distance();
            println!("map[{:?}] = {:?}", pair, distance);
            maps[pair] = distance;
        }
        assert_eq!(shortest_paths[0][1].distance(), 15.into());
        assert_eq!(shortest_paths[6][1].distance(), 110.into());
        assert_eq!(maps[vec![0, 1]], 15.into());
        assert_eq!(maps[vec![6, 1]], 110.into());
        Ok(())
    }
}
