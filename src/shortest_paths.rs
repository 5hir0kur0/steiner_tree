use crate::graph::{Edge, NodeIndex};
use crate::util::{edge, NaturalOrInfinite, PriorityValuePair};
use crate::Graph;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::iter;
use std::mem;
use std::ops::{Index, IndexMut, Range};

/// Stores a path together with its distance. Used to represent shortest paths computed by
/// [dijkstra_shortest_paths_general] or [ShortestPathMatrix::floyd_warshall].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortestPath {
    distance: NaturalOrInfinite,
    path: Vec<NodeIndex>,
}

impl ShortestPath {
    /// Create a new [ShortestPath] given the `path` and the `distance`.
    /// The `path` should not contain the start node which is instead implicitly represented by the
    /// index where the [ShortestPath] is stored.
    pub fn new(path: Vec<NodeIndex>, distance: NaturalOrInfinite) -> Self {
        Self { path, distance }
    }

    /// Create an empty [ShortestPath] with a distance of `0`.
    pub fn empty() -> Self {
        Self {
            path: vec![],
            distance: 0.into(),
        }
    }

    /// Create an empty [ShortestPath] with a distance of `NaturalOrInfinite::infinity()`.
    pub fn unreachable() -> Self {
        Self {
            path: vec![],
            distance: NaturalOrInfinite::infinity(),
        }
    }

    /// Get the distance of the path.
    pub fn distance(&self) -> NaturalOrInfinite {
        self.distance
    }

    /// Get the nodes of the path. Note that this does not contain the start node.
    pub fn path(&self) -> &[NodeIndex] {
        &self.path
    }

    /// Get the edges of the path when it starts at `start`.
    /// Edges are output in the form `(from, to)` where `from < to`.
    pub fn edges_on_path(
        &self,
        start: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, NodeIndex)> + '_ {
        iter::once(start)
            .chain(self.path().iter().copied())
            .zip(self.path().iter().copied())
            .map(|(a, b)| edge(a, b))
    }

    /// Get the edges of the path when it starts at `start`.
    /// Edges are output in the form `[from, to]` in the order in which the nodes appear on the
    /// path.
    /// **Don't treat the returned edges as undirected edges** because the rest of the code assumes
    /// that undirected edges have the form `(a, b)` where `a < b` which is not guaranteed here.
    /// To avoid confusion with undirected edges, the edges are returned as arrays instead of tuples.
    pub fn edges_on_path_directed(
        &self,
        start: NodeIndex,
    ) -> impl Iterator<Item = [NodeIndex; 2]> + '_ {
        iter::once(start)
            .chain(self.path().iter().copied())
            .zip(self.path().iter().copied())
            .map(|(a, b)| [a, b])
    }
}

impl Default for ShortestPath {
    fn default() -> Self {
        Self::unreachable()
    }
}

// ShortestPaths are ordered by their distances.
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

/// Used to store all shortest path between a subset (or all) of the nodes of a graph.
pub struct ShortestPathMatrix {
    paths: Vec<ShortestPath>,
    dimension: usize,
}

impl ShortestPathMatrix {
    /// Compute the shortest paths between all pairs of nodes in the graph.
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
        let mut terminal_to_all = vec![vec![]; graph.num_terminals()];
        for (idx, &terminal) in graph.terminals().iter().enumerate() {
            terminal_to_all[idx] = dijkstra(&graph, terminal);
        }
        for from_idx in 0..graph.num_terminals() {
            for to_idx in 0..graph.num_terminals() {
                let to = graph.terminals()[to_idx];
                result[from_idx][to_idx] = terminal_to_all[from_idx][to]
                    .take()
                    .expect("terminals not connected");
            }
        }
        result
    }

    /// The range of values where the index of the first dimension is `index`.
    fn index_range(&self, index: usize) -> Range<usize> {
        let start = index * self.dimension;
        start..start + self.dimension
    }

    /// The size of the matrix `m` in memory is `m.dimension() * m.dimension()`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// This allows for neat two-dimensional indexing (e.g. `spm[a][b]`).
impl Index<usize> for ShortestPathMatrix {
    type Output = [ShortestPath];

    fn index(&self, index: usize) -> &Self::Output {
        &self.paths[self.index_range(index)]
    }
}

/// This allows for neat two-dimensional indexing (e.g. `spm[a][b] = c`).
impl IndexMut<usize> for ShortestPathMatrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let range = self.index_range(index);
        &mut self.paths[range]
    }
}

/// Compute the shortest paths from the `start` node to all other nodes (in the same connected
/// component). Node indices must lie in the range `0..num_nodes`.
/// Path length is the sum of the edge weights as returned by `weight`. The graph is represented
/// by the adjacency function `neighbors`.
pub fn dijkstra_shortest_paths_general<W, N, I>(
    num_nodes: usize,
    weight: W,
    neighbors: N,
    start: NodeIndex,
) -> Vec<Option<ShortestPath>>
where
    W: Fn(NodeIndex, NodeIndex) -> NaturalOrInfinite,
    N: Fn(NodeIndex) -> I,
    I: Iterator<Item = NodeIndex>,
{
    // BinaryHeap is a max-heap; wrapping the items in `Reverse` effectively turns it into a min-
    // heap.
    let mut queue = BinaryHeap::new();
    queue.push(Reverse(PriorityValuePair {
        value: start,
        priority: 0.into(),
    }));
    let mut processed = HashSet::new();
    let mut parent: Vec<Option<NodeIndex>> = vec![None; num_nodes];
    let mut key = vec![NaturalOrInfinite::infinity(); num_nodes];
    key[start] = 0.into();
    while let Some(Reverse(PriorityValuePair { value: node, .. })) = queue.pop() {
        // This check is necessary because the same node might be pushed more than once; see below.
        if processed.contains(&node) {
            continue;
        }
        processed.insert(node);
        for neighbor in neighbors(node) {
            if !processed.contains(&neighbor) {
                let update = key[node] + weight(node, neighbor);
                if update < key[neighbor] {
                    // re-push instead since decrease-key is not supported
                    queue.push(Reverse(PriorityValuePair {
                        value: neighbor,
                        priority: update,
                    }));
                    key[neighbor] = update;
                    parent[neighbor] = Some(node);
                }
            }
        }
    }
    let mut shortest_paths = vec![None; num_nodes];
    for goal in 0..shortest_paths.len() {
        // trace path in reverse direction
        let path = trace_path(&parent, goal, start);
        if path.is_empty() {
            // `goal` and `start` are not connected
            continue;
        }
        let mut reversed = path;
        reversed.pop(); // remove `start`
        reversed.reverse();
        shortest_paths[goal] = Some(ShortestPath::new(reversed, key[goal]));
    }
    shortest_paths[start] = Some(ShortestPath::empty());
    shortest_paths
}

/// Helper function for Dijkstra's algorithm to track back the shortest path using the `parent`
/// array.
/// Returns an empty vector if `end` is unreachable from `start`.
fn trace_path(parent: &[Option<NodeIndex>], start: NodeIndex, end: NodeIndex) -> Vec<NodeIndex> {
    let mut res = vec![start];
    let mut current = start;
    // Just follow the "pointers" in the `parent` vector until the `end` is reached.
    while parent[current] != Some(end) {
        match parent[current] {
            Some(next) => {
                res.push(next);
                current = next;
            }
            None => return vec![],
        }
    }
    res.push(end);
    res
}

/// Run [dijkstra_shortest_paths_general] on a [Graph].
pub fn dijkstra(graph: &Graph, start: NodeIndex) -> Vec<Option<ShortestPath>> {
    dijkstra_shortest_paths_general(
        graph.num_nodes(),
        |from, to| graph.weight(from, to),
        |n| graph.neighbors(n).map(|&Edge { to, .. }| to),
        start,
    )
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
        assert_eq!(
            spm[1][6],
            ShortestPath::new(vec![2, 3, 6], (30 + 50 + 30).into())
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

    #[test]
    fn test_dijkstra() -> TestResult {
        let graph = shortcut_test_graph()?;
        let shortest_paths = dijkstra(&graph, 0);
        assert_eq!(
            shortest_paths[3],
            Some(ShortestPath::new(vec![1, 3], 3.into()))
        );
        assert_eq!(
            shortest_paths[2],
            Some(ShortestPath::new(vec![1, 2], 2.into()))
        );
        assert_eq!(
            shortest_paths[1],
            Some(ShortestPath::new(vec![1], 1.into()))
        );
        assert_eq!(shortest_paths[0], Some(ShortestPath::new(vec![], 0.into())));
        let shortest_paths = dijkstra(&graph, 2);
        assert_eq!(
            shortest_paths[3],
            Some(ShortestPath::new(vec![1, 3], 3.into()))
        );
        Ok(())
    }
}
