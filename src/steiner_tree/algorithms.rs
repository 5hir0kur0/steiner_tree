use crate::graph::{Edge, NodeIndex};
use crate::shortest_paths::ShortestPathMatrix;
use crate::steiner_tree::tree::EdgeTree;
use crate::util::{
    combinations, edge, non_trivial_subsets, sorted, NaturalOrInfinite, PriorityValuePair,
};
use crate::Graph;
use std::cmp::{min, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::iter;

// The position in the outer vector encodes the non-leaf node.
type NonLeafWeights = Vec<HashMap<Vec<NodeIndex>, NaturalOrInfinite>>;
type MinimumWeights = HashMap<Vec<NodeIndex>, NaturalOrInfinite>;

/// Dreyfus-Wagner Algorithm for finding a minimal Steiner tree.
pub fn dreyfus_wagner(graph: &Graph) -> EdgeTree {
    let shortest_paths = ShortestPathMatrix::new(graph);
    if graph.num_terminals() <= 1 {
        return EdgeTree::empty();
    }
    if graph.num_terminals() == 2 {
        return EdgeTree::new(
            &shortest_paths[graph.terminals()[0]][graph.terminals()[1]],
            graph.terminals()[0],
        );
    }
    // Indices need to be sorted in increasing order.
    let mut minimum_weights: MinimumWeights = HashMap::new();
    add_pair_steiner_trees(&mut minimum_weights, &shortest_paths, graph);
    // Subset of the S_k set in the original algorithm where k is not a leaf.
    let mut non_leaf: NonLeafWeights = vec![HashMap::new(); graph.num_nodes()];
    // You can save some time by solving it for |T|-1 terminals and then doing the last one separately.
    let terminals: &[NodeIndex] = &graph.terminals()[..graph.num_terminals() - 1];
    for subset_size in 2..terminals.len() {
        // Steiner tree sizes of size `subset_size + 1` where the additional node is not a leaf.
        calculate_non_leaf_steiner_trees(
            graph,
            terminals,
            &minimum_weights,
            &mut non_leaf,
            subset_size,
        );
        for subset in combinations(terminals, subset_size) {
            for v in nodes_not_in_subset(graph, &subset) {
                minimum_weights.insert(
                    extend_index(&subset, v),
                    weight_of_extended(
                        graph,
                        &shortest_paths,
                        &minimum_weights,
                        &non_leaf,
                        &subset,
                        v,
                    ),
                );
            }
        }
    }
    // handle the terminal we removed at the beginning
    let missing = *graph.terminals().last().expect("no terminals");
    calculate_non_leaf_steiner_trees(
        graph,
        terminals,
        &minimum_weights,
        &mut non_leaf,
        terminals.len(),
    );
    minimum_weights.insert(
        extend_index(terminals, missing),
        weight_of_extended(
            graph,
            &shortest_paths,
            &minimum_weights,
            &non_leaf,
            terminals,
            missing,
        ),
    );
    let tree = reverse_build_tree(
        graph.terminals(),
        &shortest_paths,
        &minimum_weights,
        graph,
        &non_leaf,
    );
    assert_eq!(
        minimum_weights[graph.terminals()],
        tree.weight_in(graph),
        "reconstructed tree does not have equal weight"
    );
    tree
}

/// Add Steiner trees for all pairs of nodes. The Steiner trees just consist of the shortest paths between the nodes.
fn add_pair_steiner_trees(
    minimum_weights: &mut MinimumWeights,
    shortest_paths: &ShortestPathMatrix,
    graph: &Graph,
) {
    let nodes = graph.node_indices().collect::<Vec<_>>();
    for pair in combinations(&nodes, 2) {
        let distance = shortest_paths[pair[0]][pair[1]].distance();
        debug_assert!(sorted(&pair));
        minimum_weights.insert(pair, distance);
    }
}

/// Compute the weight of the minimal Steiner tree when the terminals are extended to `subset ∪ {v}`,
/// based on existing computations of all the smaller Steiner trees.
fn weight_of_extended(
    graph: &Graph,
    shortest_paths: &ShortestPathMatrix,
    minimum_weights: &MinimumWeights,
    non_leaf: &NonLeafWeights,
    subset: &[NodeIndex],
    v: NodeIndex,
) -> NaturalOrInfinite {
    min(
        // v is connected to a node w of the tree (w in subset)
        subset
            .iter()
            .map(|&w| shortest_paths[v][w].distance())
            .min()
            .unwrap_or_else(NaturalOrInfinite::infinity)
            + minimum_weights[subset],
        // v is connected to a Steiner node (w not in subset). Note that this means that
        // w must be a inner node in the tree.
        nodes_not_in_subset(graph, subset)
            .map(|w| shortest_paths[v][w].distance() + non_leaf[w][subset])
            .min()
            .unwrap_or_else(NaturalOrInfinite::infinity),
    )
}

/// Traverse the `non_leaf` and `minimum_weights` data structures in reverse order of creation to create
/// the actual Steiner tree.
fn reverse_build_tree(
    nodes: &[NodeIndex],
    shortest_paths: &ShortestPathMatrix,
    minimum_weights: &MinimumWeights,
    graph: &Graph,
    non_leaf: &[HashMap<Vec<NodeIndex>, NaturalOrInfinite>],
) -> EdgeTree {
    if nodes.len() <= 1 {
        return EdgeTree::empty();
    }
    if nodes.len() == 2 {
        return EdgeTree::new(&shortest_paths[nodes[0]][nodes[1]], nodes[0]);
    }
    // If all nodes in `nodes` are terminals then the biggest one was added last.
    let the_one = if nodes.iter().all(|n| graph.terminals().contains(n)) {
        *nodes.last().unwrap()
    } else {
        // otherwise there is exactly one which is not a terminal
        let non_terminal = nodes
            .iter()
            .copied()
            .filter(|n| !graph.terminals().contains(n))
            .collect::<Vec<_>>();
        debug_assert_eq!(non_terminal.len(), 1);
        *non_terminal.last().unwrap()
    };
    // the nodes other than `the_one`
    let mut the_rest = nodes
        .iter()
        .copied()
        .filter(|&n| n != the_one)
        .collect::<Vec<_>>();
    the_rest.sort_unstable();
    debug_assert_eq!(the_rest.len(), nodes.len() - 1);
    // From here it is basically the same computation that was used to compute the
    // `minimum_weights`, just "in reverse" (i.e. top-down instead of bottom-up).
    for k in graph.node_indices() {
        if the_rest.binary_search(&k).is_err() {
            if shortest_paths[k][the_one].distance() + non_leaf[k][&the_rest]
                == minimum_weights[nodes]
            {
                for subset in non_trivial_subsets(&the_rest) {
                    let subset_and_k = extend_index(&subset, k);
                    let complement_and_k = extend_index_difference(&the_rest, &subset, k);
                    debug_assert_ne!(non_leaf[k][&the_rest], NaturalOrInfinite::infinity());
                    if minimum_weights[&subset_and_k] + minimum_weights[&complement_and_k]
                        == non_leaf[k][&the_rest]
                    {
                        let tree1 = reverse_build_tree(
                            &subset_and_k,
                            shortest_paths,
                            minimum_weights,
                            graph,
                            non_leaf,
                        );
                        if tree1.is_empty() {
                            continue;
                        }
                        let tree2 = reverse_build_tree(
                            &complement_and_k,
                            shortest_paths,
                            minimum_weights,
                            graph,
                            non_leaf,
                        );
                        if tree2.is_empty() {
                            continue;
                        }
                        let mut tree = EdgeTree::new(&shortest_paths[k][the_one], k);
                        tree.extend(&tree1);
                        tree.extend(&tree2);
                        return tree;
                    }
                }
            }
        } else {
            if shortest_paths[k][the_one].distance() + minimum_weights[&the_rest]
                == minimum_weights[nodes]
            {
                let mut tree =
                    reverse_build_tree(&the_rest, shortest_paths, minimum_weights, graph, non_leaf);
                if !tree.is_empty() {
                    tree.extend(&EdgeTree::new(&shortest_paths[k][the_one], k));
                    return tree;
                }
            }
        }
    }
    return EdgeTree::empty();
}

/// Calculate minimal Steiner trees of the node sets `T ∪ {v}`
/// where `T ⊆ graph.terminals(), |T| = subset_size` and `v ∈ graph.node_indices() \ T` in which
/// `v` is **not a leaf**.
fn calculate_non_leaf_steiner_trees(
    graph: &Graph,
    terminals: &[NodeIndex],
    minimum_weights: &MinimumWeights,
    non_leaf: &mut NonLeafWeights,
    subset_size: usize,
) {
    for subset in combinations(terminals, subset_size) {
        for v in nodes_not_in_subset(graph, &subset) {
            debug_assert!(sorted(&subset));
            non_leaf[v].insert(
                subset.clone(),
                non_trivial_subsets(&subset)
                    // Since we're always adding the value of the set and the one of its complement
                    // the values we get for choosing the set itself and its complement as
                    // `sub_subset` are equal. Because of the way subsets are enumerated by
                    // `non_trivial_subsets()` it suffices to take the first half of the values.
                    .take((2_usize.pow(subset.len() as u32) - 2) / 2)
                    .map(|sub_subset| {
                        let sub_subset_and_v = extend_index(&sub_subset, v);
                        let sub_subset_complement_and_v =
                            extend_index_difference(&subset, &sub_subset, v);
                        minimum_weights[&sub_subset_and_v]
                            + minimum_weights[&sub_subset_complement_and_v]
                    })
                    .min()
                    .unwrap_or_else(NaturalOrInfinite::infinity),
            );
        }
    }
}

/// Calculate `graph.node_indices() \ subset`.
/// `subset` needs to be sorted.
///
/// # Panics (debug builds)
/// If `subset` is not sorted.
fn nodes_not_in_subset<'a, 'b>(
    graph: &'a Graph,
    subset: &'b [NodeIndex],
) -> impl Iterator<Item = NodeIndex> + 'b
where
    'a: 'b,
{
    debug_assert!(sorted(&subset));
    graph
        .node_indices()
        .filter(move |v| subset.binary_search(v).is_err())
}

/// Calculate `old_index ∪ {element}`, cloning `old_index`.
/// Requires that `element ∉ old_index`.
fn extend_index(old_index: &[NodeIndex], element: NodeIndex) -> Vec<NodeIndex> {
    let mut res = old_index.to_vec();
    debug_assert!(!old_index.contains(&element));
    res.push(element);
    res.sort_unstable();
    res
}

/// Calculate `(set \ other) ∪ {element}`, cloning `old_index`.
/// Requires that `element ∉ set` and `other` to be sorted.
fn extend_index_difference(
    set: &[NodeIndex],
    other: &[NodeIndex],
    element: NodeIndex,
) -> Vec<NodeIndex> {
    let mut complement_and_el = set.to_vec();
    debug_assert!(sorted(other));
    debug_assert!(!set.contains(&element));
    complement_and_el.retain(|e| other.binary_search(e).is_err());
    complement_and_el.push(element);
    complement_and_el.sort_unstable();
    complement_and_el
}

pub fn kou_et_al_steiner_approximation(graph: &Graph) -> EdgeTree {
    // Construct a complete graph from the steiner points (edge weights = distances in original graph).
    let complete_edges = ShortestPathMatrix::terminal_distances(graph);
    // Compute its minimum spanning tree.
    let complete_mst = prim_spanning_tree(
        complete_edges.dimension(),
        |from, to| complete_edges[from][to].distance(),
        |n| (0..complete_edges.dimension()).filter(move |&x| x != n),
        0,
    );
    // Construct a subgaph of the original graph (represented here as a set of
    // edges) that contains all nodes/edges on the original shortest paths between the nodes in
    // the MST above.
    let mut subgraph = HashSet::new();
    for &(from_idx, to_idx) in complete_mst.edges() {
        let (from, to) = (graph.terminals()[from_idx], graph.terminals()[to_idx]);
        #[cfg(debug_assertions)]
        {
            let path_edges = complete_edges[from_idx][to_idx]
                .edges_on_path(from)
                .collect::<Vec<_>>();
            // could be in .0 or .1 since edges are undirected (and therefore represented by ordered tuples)
            debug_assert!(path_edges[0].0 == from || path_edges[0].1 == from);
            let last = path_edges.last().unwrap();
            debug_assert!(last.1 == to || last.0 == to);
        }
        complete_edges[from_idx][to_idx]
            .edges_on_path(from)
            .for_each(|edge| {
                subgraph.insert(edge);
            });
    }
    // let subgraph_nodes = subgraph
    //     .iter()
    //     .flat_map(|&(a, b)| iter::once(a).chain(iter::once(b)))
    //     .collect::<HashSet<_>>();

    // Compute a MST of the subgraph.
    let mut subgraph_mst = prim_spanning_tree(
        graph.num_nodes(),
        |from, to| {
            if subgraph.contains(&edge(from, to)) {
                graph.weight(from, to)
            } else {
                NaturalOrInfinite::infinity()
            }
        },
        |n| {
            let subgraph = &subgraph;
            graph
                .neighbors(n)
                .map(|&Edge { to, .. }| to)
                .filter(move |to| subgraph.contains(&edge(n, *to)))
        },
        graph.terminals()[0],
    );
    // If there are leaves that are non-terminal nodes then the weight can be
    // decreased by removing the respective edges.
    remove_non_terminal_leaves(&mut subgraph_mst, graph);
    subgraph_mst
}

/// Removes paths in the tree that lead to leaves which are not terminal nodes.
/// After this function returns, all leaves of the tree are terminal nodes.
fn remove_non_terminal_leaves(subgraph_mst: &mut EdgeTree, graph: &Graph) {
    let mut encountered: HashMap<NodeIndex, (u32, (NodeIndex, NodeIndex))> = HashMap::new();
    let mut non_terminal_leaf_found = true;
    while non_terminal_leaf_found {
        encountered.clear();
        for &(from, to) in subgraph_mst.edges() {
            let edge = (from, to);
            for &n in &[from, to] {
                encountered.entry(n).or_insert((0, edge)).0 += 1;
            }
        }
        non_terminal_leaf_found = false;
        for node in encountered.keys() {
            let (count, edge) = encountered[node];
            if count == 1 && !graph.terminals().contains(node) {
                non_terminal_leaf_found = true;
                subgraph_mst.remove(edge);
            }
        }
    }
}

/// Construct a minimum spanning tree of the nodes reachable from `start`. The edge weights are
/// taken from the `weight` function and the neighbors of each node are found by using the
/// `neighbors` function.
/// This function requires that the nodes lie in the range `0..num_nodes`.
fn prim_spanning_tree<W, N, I>(
    num_nodes: usize,
    weight: W,
    neighbors: N,
    start: NodeIndex,
) -> EdgeTree
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
    while let Some(Reverse(PriorityValuePair { value: node, .. })) = queue.pop() {
        if processed.contains(&node) {
            continue;
        }
        processed.insert(node);
        for neighbor in neighbors(node) {
            if !processed.contains(&neighbor) {
                let update = weight(node, neighbor);
                if update < key[neighbor] {
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
    let mut tree = EdgeTree::empty();
    for (node, parent_node) in parent.iter().enumerate() {
        if let &Some(parent_node) = parent_node {
            assert_ne!(node, parent_node);
            let edge = edge(node, parent_node);
            tree.insert(edge);
        }
    }
    assert_eq!(tree.edges().len(), processed.len() - 1);
    tree
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::graph::tests::{
        diamond_test_graph, shortcut_test_graph, small_test_graph, steiner_example_paper,
        steiner_example_wiki,
    };
    use crate::util::TestResult;

    fn assert_contains_terminals(tree: &EdgeTree, terminals: &[NodeIndex]) {
        let nodes = tree.nodes();
        assert!(
            terminals.iter().all(|t| nodes.contains(t)),
            "tree {:?} does not contain all terminals ({:?})",
            tree,
            terminals
        );
    }

    fn assert_leaves_are_terminals(tree: &EdgeTree, terminals: &[NodeIndex]) {
        let leaves = tree.find_leaves();
        for leaf in leaves {
            assert!(terminals.contains(&leaf));
        }
    }

    fn assert_tree(tree: &EdgeTree) {
        assert_eq!(tree.edges().len(), tree.nodes().len() - 1);
        let mut stack = vec![*tree.nodes().iter().next().unwrap()];
        let mut found = HashSet::new();
        while let Some(top) = stack.pop() {
            for neighbor in tree.neighbors(top) {
                if !found.contains(&neighbor) {
                    stack.push(neighbor);
                    found.insert(neighbor);
                }
            }
        }
        assert_eq!(found, tree.nodes());
    }

    fn assert_plausible_steiner_tree(tree: &EdgeTree, graph: &Graph) {
        assert_contains_terminals(tree, graph.terminals());
        assert_leaves_are_terminals(tree, graph.terminals());
        assert_tree(tree);
    }

    #[test]
    fn test_dreyfus_wagner_trivial() -> TestResult {
        let trivial = small_test_graph()?;
        let result = dreyfus_wagner(&trivial);
        assert_eq!(result.weight_in(&trivial), 3.into());
        assert_plausible_steiner_tree(&result, &trivial);
        Ok(())
    }

    #[test]
    fn test_dreyfus_wagner_wiki_example() -> TestResult {
        let graph = steiner_example_wiki()?;
        let result = dreyfus_wagner(&graph);
        assert_eq!(
            result.weight_in(&graph),
            NaturalOrInfinite::from(25 + 30 + 15 + 10 + 40 + 50 + 20)
        );
        assert_plausible_steiner_tree(&result, &graph);
        assert_eq!(
            result.edges(),
            &[(0, 4), (4, 8), (8, 10), (10, 11), (9, 10), (7, 9), (6, 7),]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_dreyfus_wagner_diamond() -> TestResult {
        let graph = diamond_test_graph()?;
        let tree = dreyfus_wagner(&graph);
        assert_plausible_steiner_tree(&tree, &graph);
        assert_eq!(tree.weight_in(&graph), 7.into());
        assert_eq!(
            tree.edges(),
            &[(0, 1), (1, 3), (3, 4), (4, 5)]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_dreyfus_wagner_paper_example() -> TestResult {
        let graph = steiner_example_paper()?;
        let tree = dreyfus_wagner(&graph);
        assert_eq!(tree.weight_in(&graph), 5.into());
        assert_plausible_steiner_tree(&tree, &graph);
        Ok(())
    }

    #[test]
    fn vec_hash() {
        let mut v1 = Vec::with_capacity(100);
        let mut v2 = vec![];
        v1.push(42);
        v2.push(42);
        let mut map = HashMap::new();
        map.insert(v1, "hi");
        assert_eq!(map[&v2], "hi");
    }

    fn prim_mst(graph: &Graph) -> EdgeTree {
        prim_spanning_tree(
            graph.num_nodes(),
            |from, to| graph.weight(from, to),
            |n| graph.neighbors(n).map(|&Edge { to, .. }| to),
            0,
        )
    }

    #[test]
    fn test_prim() -> TestResult {
        let graph = small_test_graph()?;
        let tree = prim_mst(&graph);
        assert_eq!(tree.weight_in(&graph), 3.into());
        assert_eq!(
            tree.edges(),
            &[(0, 1), (1, 2),].iter().copied().collect::<HashSet<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_prim_2() -> TestResult {
        let graph = shortcut_test_graph()?;
        let tree = prim_mst(&graph);
        assert_eq!(tree.weight_in(&graph), 4.into());
        assert_eq!(
            tree.edges(),
            &[(0, 1), (1, 2), (1, 3)]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_kou_approximation() -> TestResult {
        let graphs = [
            small_test_graph()?,
            shortcut_test_graph()?,
            steiner_example_paper()?,
            steiner_example_wiki()?,
        ];
        for graph in &graphs {
            let exact = dreyfus_wagner(graph);
            let approx = kou_et_al_steiner_approximation(graph);
            assert_plausible_steiner_tree(&exact, &graph);
            assert_plausible_steiner_tree(&approx, &graph);
            let leaves = exact.find_leaves().len();
            let exact_weight = exact.weight_in(graph).finite_value();
            let approx_weight = approx.weight_in(graph).finite_value();
            println!(
                "exact_weight = {}, approx_weight = {}, leaves = {:?}",
                exact_weight, approx_weight, leaves
            );
            assert!(
                (approx_weight as f64) / (exact_weight as f64) <= (2.0 - (2.0 / (leaves as f64)))
            );
        }
        Ok(())
    }

    #[test]
    fn test_count_leaves() {
        let mut tree = EdgeTree::empty();
        assert!(tree.find_leaves().is_empty());
        tree.insert((0, 1));
        assert_eq!(
            tree.find_leaves(),
            [0, 1].iter().copied().collect::<HashSet<_>>()
        );
        tree.insert((1, 2));
        assert_eq!(
            tree.find_leaves(),
            [0, 2].iter().copied().collect::<HashSet<_>>()
        );
        tree.insert((1, 3));
        assert_eq!(
            tree.find_leaves(),
            [0, 2, 3].iter().copied().collect::<HashSet<_>>()
        );
        tree.insert((2, 3));
        assert_eq!(
            tree.find_leaves(),
            [0].iter().copied().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn test_remove_non_terminal_leaves() -> TestResult {
        let graph = steiner_example_wiki()?;
        let mut tree = EdgeTree::empty();
        [(0, 4), (4, 8), (8, 10), (10, 11), (9, 10), (7, 9), (6, 7),
         // unnecessary edges:
         (5, 7),
         (0, 1),
         (1, 2),
         (2, 3),
        ].iter().for_each(|&edge| tree.insert(edge));
        remove_non_terminal_leaves(&mut tree, &graph);
        assert_eq!(
            tree.edges(),
            &[(0, 4), (4, 8), (8, 10), (10, 11), (9, 10), (7, 9), (6, 7),]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
        );
        Ok(())
    }
}
