use std::cmp::Ordering;
use std::error::Error;
use std::ops::{Add, Index, IndexMut};

/// Result with boxed error as trait object.
pub type GenericResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[cfg(test)]
pub(crate) type TestResult = GenericResult<()>;

/// Numerical type that can either be an unsigned number or positive infinity.
// infinity is encoded as `-1`; all other negative values are illegal
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NaturalOrInfinite(i64);

impl PartialOrd for NaturalOrInfinite {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0 < 0 && other.0 < 0 {
            Some(Ordering::Equal)
        } else if other.0 < 0 {
            Some(Ordering::Less)
        } else if self.0 < 0 {
            Some(Ordering::Greater)
        } else {
            Some(self.0.cmp(&other.0))
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
        NaturalOrInfinite(val as i64)
    }
}

/// Return all combinations (subsets) of the given length.
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

pub struct Combinations<'a, T: Copy> {
    indices: Vec<usize>,
    elements: &'a [T],
    count_position: usize,
    buffer: Vec<T>,
    first_iteration: bool,
}

impl<'a, T: Copy> Combinations<'a, T> {
    fn index_done(&self, index: usize) -> bool {
        self.indices[index] >= self.elements.len() - (self.indices.len() - index)
    }

    fn update_buffer(&mut self, index: usize) {
        self.buffer[index] = self.elements[self.indices[index]];
    }

    fn increment(&mut self) -> bool {
        if !self.index_done(self.count_position) {
            self.indices[self.count_position] += 1;
            self.update_buffer(self.count_position);
        } else {
            let mut i = self.count_position;
            while self.index_done(i) {
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

    // I can't make this into a "real" iterator because the iterator protocol is too restrictive
    // to allow returning references to the iterator itself so I would have to allocate each time
    // which I don't want to.
    pub(crate) fn next(&mut self) -> Option<&[T]> {
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

    #[cfg(test)]
    fn collect(&mut self) -> Vec<Vec<T>> {
        let mut res = vec![];
        while let Some(val) = self.next() {
            res.push(val.to_vec());
        }
        res
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
        let vec = vec![T::default(); size];
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
        debug_assert!(index
            .iter()
            .enumerate()
            .all(|(ii, &i)| i < self.dimensions[ii]));
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
        // indices {1, 2, 3} can never have 1 as its second index.
        let dimensions = (0..subset_size).map(|i| num_elements - i).collect();
        Self {
            map: Vector::new(dimensions),
            subset_size,
        }
    }

    fn calculate_index(index: &mut [usize]) {
        index.sort_unstable();
        index.iter_mut().enumerate().map(|(ii, i)| *i -= ii);
        #[cfg(debug_assertions)]
        {
            let mut index2 = index.to_vec();
            index2.dedup();
            debug_assert!(index2.len() == index.len())
        }
    }
}

impl<T: Copy + Default> Index<&mut [usize]> for IndexSetMap<T> {
    type Output = T;

    fn index(&self, index: &mut [usize]) -> &Self::Output {
        Self::calculate_index(index);
        &self.map[index]
    }
}

impl<T: Copy + Default> IndexMut<&mut [usize]> for IndexSetMap<T> {
    fn index_mut(&mut self, index: &mut [usize]) -> &mut Self::Output {
        Self::calculate_index(index);
        &mut self.map[index]
    }
}

impl<T: Copy + Default> Index<Vec<usize>> for IndexSetMap<T> {
    type Output = T;

    fn index(&self, mut index: Vec<usize>) -> &Self::Output {
        self.index(&mut index[..])
    }
}

impl<T: Copy + Default> IndexMut<Vec<usize>> for IndexSetMap<T> {
    fn index_mut(&mut self, mut index: Vec<usize>) -> &mut Self::Output {
        self.index_mut(&mut index[..])
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(combinations(empty, 0).collect(), vec![[]]);
        let one = &[1];
        assert_eq!(combinations(one, 0).collect(), vec![[]]);
        assert_eq!(combinations(one, 1).collect(), vec![[1]]);
        let two = &[1, 2];
        assert_eq!(combinations(two, 0).collect(), vec![[]]);
        assert_eq!(combinations(two, 1).collect(), vec![[1], [2]]);
        assert_eq!(combinations(two, 2).collect(), vec![[1, 2]]);
    }

    #[test]
    fn test_combinations() {
        let four = &[1, 2, 3, 4];
        assert_eq!(combinations(four, 0).collect(), vec![[]]);
        assert_eq!(combinations(four, 1).collect(), vec![[1], [2], [3], [4]]);
        assert_eq!(
            combinations(four, 2).collect(),
            vec![[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        );
        assert_eq!(
            combinations(four, 3).collect(),
            vec![[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        );
        assert_eq!(combinations(four, 4).collect(), vec![[1, 2, 3, 4]]);
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
        let mut vec: Vector<&str> = Vector::new(vec![12, 24, 7, 17, 13, 5]);
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
    fn test_index_set_map() {
        let mut map = IndexSetMap::new(100, 3);
        map[vec![1, 2, 3]] = "frog";
        let mut index = vec![3, 1, 2];
        assert_eq!(map[&mut index[..]], "frog");
        assert_eq!(index, vec![1, 2, 3]);
        map[&mut [42usize, 21usize, 10usize][..]] = "dragonfly";
        assert_eq!(map[&mut [10usize, 21usize, 42usize][..]], "dragonfly");
    }
}
