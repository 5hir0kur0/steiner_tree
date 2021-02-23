use crate::util::NaturalOrInfinite;
use crate::Graph;
use std::ops::{Index, IndexMut, Range};

pub struct ShortestPathMatrix {
    paths: Vec<NaturalOrInfinite>,
    dimension: usize,
}

impl ShortestPathMatrix {
    pub fn new(graph: &Graph) -> Self {
        let n = graph.number_of_nodes();
        let paths = vec![NaturalOrInfinite::infinity(); n * n];
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
            self[from][to] = weight.into();
            self[to][from] = weight.into();
        }
        for n in graph.node_indices() {
            self[n][n] = NaturalOrInfinite::from(0);
        }
        for k in graph.node_indices() {
            for i in graph.node_indices() {
                for j in graph.node_indices() {
                    let new_dist = self[i][k] + self[k][j];
                    if new_dist < self[i][j] {
                        self[i][j] = new_dist;
                    }
                }
            }
        }
    }

    fn index_range(&self, index: usize) -> Range<usize> {
        let start = index * self.dimension;
        start..start + self.dimension
    }
}

/// This allows for neat two-dimensional indexing (e.g. `spa[a][b]`).
impl Index<usize> for ShortestPathMatrix {
    type Output = [NaturalOrInfinite];

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
    use crate::graph::tests::{shortcut_test_graph, small_test_graph};
    use crate::util::TestResult;

    #[test]
    fn test_shortest_path_matrix() -> TestResult {
        let graph = small_test_graph()?;
        let spm = ShortestPathMatrix::new(&graph);
        assert_eq!(spm[0][1], 1.into());
        assert_eq!(spm[1][2], 2.into());
        assert_eq!(spm[0][2], 3.into());
        Ok(())
    }

    #[test]
    fn test_shortest_path_matrix2() -> TestResult {
        let graph = shortcut_test_graph()?;
        let spm = ShortestPathMatrix::new(&graph);
        assert_eq!(spm[0][2], 2.into());
        assert_eq!(spm[3][0], 3.into());
        assert_eq!(spm[3][2], 3.into());

        for i in 0..spm.dimension {
            for j in 0..spm.dimension {
                assert_eq!(spm[i][j], spm[j][i]);
            }
        }
        Ok(())
    }
}
