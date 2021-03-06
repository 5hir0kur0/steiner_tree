use crate::graph::NodeIndex;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Add;

#[cfg(test)]
pub(crate) type TestResult = Result<(), Box<dyn std::error::Error>>;

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

    /// Adding something to `infinity` will result in `infinity`.
    fn add(self, rhs: Self) -> Self::Output {
        if self.0 < 0 || rhs.0 < 0 {
            NaturalOrInfinite(-1)
        } else {
            NaturalOrInfinite(self.0 + rhs.0)
        }
    }
}

impl NaturalOrInfinite {
    /// The value `infinity()` is bigger than all other values of [NaturalOrInfinite] except
    /// `infinity()` itself.
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

/// Iterator over the combinations of the elements of a given set that have a specified length.
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

    /// Return a pointer to the combination that will be returned and go the next combination.
    /// This can be used to iterate over the combinations without needing an allocation each time.
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

/// Return an iterator over all nontrivial (i.e. not the empty set and not the full set) subsets.
/// The sets are ordered by their length.
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

/// Check if the slice is sorted in non-decreasing order..
pub(crate) fn is_sorted<T: Ord>(elements: &[T]) -> bool {
    for (a, b) in elements.iter().zip(elements.iter().skip(1)) {
        if a > b {
            return false;
        }
    }
    true
}

pub(crate) fn is_sorted_by_key<T, F: Fn(&T) -> K, K: Ord>(elements: &[T], key: F) -> bool {
    for (a, b) in elements.iter().zip(elements.iter().skip(1)) {
        if key(a) > key(b) {
            return false;
        }
    }
    true
}

/// Stores a `value` together with a `priority`. Can be used for priority queues such as
/// [std::collections::BinaryHeap].
/// The values will be ordered by the `priority`.
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

/// Returns a tuple `(a, b)` such that `a < b`.
pub fn edge(from: NodeIndex, to: NodeIndex) -> (NodeIndex, NodeIndex) {
    assert_ne!(from, to);
    (from.min(to), from.max(to))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

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
                assert!(is_sorted(&comb))
            }
        }
    }
}
