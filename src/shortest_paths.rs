use crate::graph::{Edge, NodeIndex};
use crate::util::NaturalOrInfinite;
use crate::Graph;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter;
use std::mem;
use std::ops::{Index, IndexMut, Range};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortestPath {
    distance: NaturalOrInfinite,
    path: Vec<NodeIndex>,
}

impl ShortestPath {
    pub fn new(path: Vec<NodeIndex>, distance: NaturalOrInfinite) -> Self {
        Self { path, distance }
    }

    pub fn empty() -> Self {
        Self {
            path: vec![],
            distance: 0.into(),
        }
    }

    pub fn distance(&self) -> NaturalOrInfinite {
        self.distance
    }

    pub fn path(&self) -> &[NodeIndex] {
        &self.path
    }

    /// Edges are output in the form `(from, to)` where `from < to`.
    pub fn edges_on_path(
        &self,
        start: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, NodeIndex)> + '_ {
        iter::once(start)
            .chain(self.path().iter().copied())
            .zip(self.path().iter().copied())
            .map(|(a, b)| (a.min(b), a.max(b)))
    }
}

impl Default for ShortestPath {
    fn default() -> Self {
        Self {
            distance: NaturalOrInfinite::infinity(),
            path: vec![],
        }
    }
}

impl Ord for ShortestPath {
    fn cmp(&self, other: &Self) -> Ordering {
        // compare paths as well for consistency with PartialEq/Eq
        self.distance()
            .cmp(&other.distance())
            .then_with(|| self.path.cmp(&other.path))
    }
}

impl PartialOrd for ShortestPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct ShortestPathMatrix {
    paths: Vec<ShortestPath>,
    dimension: usize,
}

impl ShortestPathMatrix {
    pub fn new(graph: &Graph) -> Self {
        let n = graph.num_nodes();
        let paths = vec![ShortestPath::default(); n * n];
        let mut res = ShortestPathMatrix {
            paths,
            dimension: n,
        };
        res.floyd_warshall(graph);
        res
    }

    /// Based on the pseudo-code
    /// [on Wikipedia](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm).
    fn floyd_warshall(&mut self, graph: &Graph) {
        for (from, to, weight) in graph.edges() {
            self[from][to] = ShortestPath::new(vec![to], weight.into());
            self[to][from] = ShortestPath::new(vec![from], weight.into());
        }
        for n in graph.node_indices() {
            self[n][n] = ShortestPath::new(vec![], 0.into());
        }
        for k in graph.node_indices() {
            for i in graph.node_indices() {
                for j in graph.node_indices() {
                    let new_dist = self[i][k].distance() + self[k][j].distance();
                    if new_dist < self[i][j].distance() {
                        self[i][j].distance = new_dist;
                        let mut ij = mem::take(&mut self[i][j].path);
                        ij.clear();
                        ij.extend_from_slice(self[i][k].path());
                        ij.extend_from_slice(&self[k][j].path());
                        self[i][j].path = ij;
                        debug_assert!(self[i][j].path.ends_with(&[j]));
                    }
                }
            }
        }
    }

    /// Returns a [ShortestPathMatrix] with the distances between all the terminal nodes of the
    /// graph.
    pub fn terminal_distances(graph: &Graph) -> Self {
        let n = graph.num_terminals();
        let mut result = Self {
            paths: vec![ShortestPath::default(); n * n],
            dimension: n,
        };
        // This could be done more efficiently as we don't actually need the shortest paths for
        // all pairs.
        let spm = Self::new(graph);
        for from_idx in 0..graph.num_terminals() {
            for to_idx in 0..graph.num_terminals() {
                let from = graph.terminals()[from_idx];
                let to = graph.terminals()[to_idx];
                result[from_idx][to_idx] = spm[from][to].clone();
            }
        }
        result
    }

    fn index_range(&self, index: usize) -> Range<usize> {
        let start = index * self.dimension;
        start..start + self.dimension
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn edges_on_path(
        &self,
        from: NodeIndex,
        to: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, NodeIndex)> + '_ {
        self[from][to].edges_on_path(from)
    }
}

/// This allows for neat two-dimensional indexing (e.g. `spa[a][b]`).
impl Index<usize> for ShortestPathMatrix {
    type Output = [ShortestPath];

    fn index(&self, index: usize) -> &Self::Output {
        &self.paths[self.index_range(index)]
    }
}

/// This allows for neat two-dimensional indexing (e.g. `spa[a][b] = c`).
impl IndexMut<usize> for ShortestPathMatrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let range = self.index_range(index);
        &mut self.paths[range]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::tests::{shortcut_test_graph, small_test_graph, steiner_example_wiki};
    use crate::util::TestResult;
    use std::iter;

    #[test]
    fn test_shortest_path_matrix_1() -> TestResult {
        let graph = small_test_graph()?;
        let spm = ShortestPathMatrix::new(&graph);
        assert_eq!(spm[0][1], ShortestPath::new(vec![1], 1.into()));
        assert_eq!(spm[1][2], ShortestPath::new(vec![2], 2.into()));
        assert!(
            spm[0][2] == ShortestPath::new(vec![2], 3.into())
                || spm[0][2] == ShortestPath::new(vec![1, 2], 3.into())
        );
        Ok(())
    }

    fn assert_paths_equiv(spm: &ShortestPathMatrix) {
        for i in 0..spm.dimension {
            for j in 0..spm.dimension {
                assert_eq!(spm[i][j].distance(), spm[j][i].distance());
                assert_eq!(
                    iter::once(i)
                        .chain(spm[i][j].path().iter().copied())
                        .rev()
                        .collect::<Vec<_>>(),
                    iter::once(j)
                        .chain(spm[j][i].path().iter().copied())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn test_shortest_path_matrix_2() -> TestResult {
        let graph = shortcut_test_graph()?;
        let spm = ShortestPathMatrix::new(&graph);
        assert_eq!(spm[0][2], ShortestPath::new(vec![1, 2], 2.into()));
        assert_eq!(spm[3][0], ShortestPath::new(vec![1, 0], 3.into()));
        assert_eq!(spm[3][2], ShortestPath::new(vec![1, 2], 3.into()));

        assert_paths_equiv(&spm);
        Ok(())
    }

    #[test]
    fn test_shortest_path_matrix_3() -> TestResult {
        let graph = steiner_example_wiki()?;
        let spm = ShortestPathMatrix::new(&graph);
        assert_eq!(
            spm[11][0],
            ShortestPath::new(vec![10, 8, 4, 0], (10 + 15 + 30 + 25).into())
        );
        assert_eq!(spm[6][9], ShortestPath::new(vec![7, 9], (50 + 20).into()));
        assert_eq!(
            spm[6][11],
            ShortestPath::new(vec![7, 9, 10, 11], (10 + 40 + 50 + 20).into())
        );
        assert_eq!(
            spm[6][0],
            ShortestPath::new(vec![3, 2, 1, 0], (30 + 50 + 30 + 15).into())
        );
        assert_paths_equiv(&spm);
        Ok(())
    }

    #[test]
    fn test_terminal_distances() -> TestResult {
        let graph = small_test_graph()?;
        let dist = ShortestPathMatrix::terminal_distances(&graph);
        assert_eq!(dist[0][0], ShortestPath::empty());
        assert_eq!(dist[1][1], ShortestPath::empty());
        assert!(
            dist[0][1] == ShortestPath::new(vec![2], 3.into())
                || dist[0][1] == ShortestPath::new(vec![1, 2], 3.into())
        );
        assert!(
            dist[1][0] == ShortestPath::new(vec![0], 3.into())
                || dist[1][0] == ShortestPath::new(vec![1, 0], 3.into())
        );
        assert_eq!(dist.dimension(), 2);
        Ok(())
    }

    #[test]
    fn test_terminal_distances_2() -> TestResult {
        let graph = shortcut_test_graph()?;
        let dist = ShortestPathMatrix::terminal_distances(&graph);
        assert_eq!(dist.dimension(), 2);
        assert_eq!(dist[0][0], ShortestPath::empty());
        assert_eq!(dist[1][1], ShortestPath::empty());
        assert_eq!(dist[0][1], ShortestPath::new(vec![1, 2], 2.into()));
        assert_eq!(dist[1][0], ShortestPath::new(vec![1, 0], 2.into()));
        Ok(())
    }
}
