use crate::graph::NodeIndex;
use crate::shortest_paths::{ShortestPath, ShortestPathMatrix};
use crate::steiner_tree::tree::EdgeTree;
use crate::util::{
    combinations, non_trivial_subsets, sorted, IndexSetMap, IndexSetMaps, NaturalOrInfinite,
};
use crate::Graph;
use std::cmp::min;
use std::collections::{HashMap, HashSet};

/// Dreyfus-Wagner Algorithm for finding a minimal Steiner tree.
pub fn dreyfus_wagner(graph: &Graph) -> NaturalOrInfinite {
    let shortest_paths = ShortestPathMatrix::new(graph);
    // the index into maps has to be offset by 2 since the map with subset-size 2 is at index 0
    let mut maps: IndexSetMaps<NaturalOrInfinite> = IndexSetMaps::new(2);
    // Subset of the S_k set in the original algorithm where k is not a leaf.
    let mut non_leaf = vec![HashMap::new(); graph.num_nodes()];
    maps.push(IndexSetMap::new(graph.num_nodes(), 2));
    let nodes = graph.node_indices().collect::<Vec<_>>();
    for pair in combinations(&nodes, 2) {
        let distance = shortest_paths[pair[0]][pair[1]].distance();
        maps[pair] = distance;
    }
    for subset_size in 2..graph.num_terminals() {
        // Steiner tree sizes of size `subset_size + 1` where the additional node is not a leaf.
        calculate_non_leaf_steiner_trees(graph, &maps, &mut non_leaf, subset_size);
        maps.push(IndexSetMap::new(graph.num_nodes(), subset_size + 1));
        for subset in combinations(graph.terminals(), subset_size) {
            for v in nodes_not_in_subset(graph, &subset) {
                maps[extend_index(&subset, v)] = min(
                    // v is connected to a node w of the tree (w in subset)
                    subset
                        .iter()
                        .map(|&w| {
                            println!("shortest_paths[{v}][{w}].distance() + maps[{subset:?}] = {shortest:?} + {maps:?} = {total:?}", shortest=shortest_paths[v][w].distance(), maps=maps[subset.to_vec()], v=v, w=w, subset=&subset, total=shortest_paths[v][w].distance() + maps[subset.to_vec()]);
                            shortest_paths[v][w].distance()
                        })
                        .min()
                        .unwrap_or_else(NaturalOrInfinite::infinity)
                        + maps[subset.to_vec()],
                    // v is connected to a Steiner node (w not in subset). Note that this means that
                    // w must be a inner node in the tree.
                    nodes_not_in_subset(graph, &subset)
                        .map(|w| {
                            println!("shortest_paths[{v}][{w}].distance() + sk[{w}][{subset:?}] = {total:?}",
                                     total=shortest_paths[v][w].distance() + non_leaf[w][&subset],
                                     v=v,
                                     w=w,
                                     subset=&subset,
                            );
                            shortest_paths[v][w].distance() + non_leaf[w][&subset]
                        })
                        .min()
                        .unwrap_or_else(NaturalOrInfinite::infinity),
                );
            }
        }
    }
    let tree = reverse_generate_tree(
        dbg!(graph.terminals()),
        &shortest_paths,
        &maps,
        graph,
        &non_leaf,
    );
    let weight = tree.weight_in(graph);
    println!(
        "=========== tree weight: {:?} =========== maps weight: {:?} ============",
        weight,
        maps[graph.terminals().to_vec()]
    );
    weight
}

/// Traverse the `non_leaf` and `maps` data structures in reverse order of creation to create
/// the actual Steiner tree.
fn reverse_generate_tree(
    nodes: &[NodeIndex],
    shortest_paths: &ShortestPathMatrix,
    maps: &IndexSetMaps<NaturalOrInfinite>,
    graph: &Graph,
    non_leaf: &[HashMap<Vec<NodeIndex>, NaturalOrInfinite>],
) -> EdgeTree {
    if nodes.len() <= 1 {
        return EdgeTree::empty();
    }

    let k = nodes[nodes.len() - 1];

    let without_k = &nodes[..nodes.len() - 1];
    let mut min_edges = EdgeTree::empty();
    let mut min_weight = NaturalOrInfinite::infinity();
    let mut found_using_loop_1 = false;
    let mut loop_2_x = None;

    if without_k.len() == 1 {
        let x = without_k[0];
        return EdgeTree::new(&shortest_paths[x][k], x);
    }

    for &x in without_k {
        let x_weight = shortest_paths[x][k].distance() + maps[without_k.to_vec()];
        if x_weight < min_weight {
            min_weight = x_weight;
            min_edges = EdgeTree::new(&shortest_paths[x][k], x);
            found_using_loop_1 = true;
        }
    }

    for x in nodes_not_in_subset(graph, without_k) {
        let x_weight = shortest_paths[x][k].distance() + non_leaf[x][&without_k.to_vec()];
        if x_weight < min_weight {
            min_weight = x_weight;
            min_edges = EdgeTree::new(&shortest_paths[x][k], x);
            found_using_loop_1 = false;
            loop_2_x = Some(x);
        }
    }

    if found_using_loop_1 {
        let rest = reverse_generate_tree(without_k, shortest_paths, maps, graph, non_leaf);
        min_edges.extend(&rest);
    } else {
        let x = loop_2_x.unwrap();
        for subset in non_trivial_subsets(&without_k) {
            let mut set2 = without_k.to_vec();
            set2.retain(|e| !subset.contains(e));
            set2.push(x);
            let mut set1 = subset;
            set1.push(x);
            if maps[set1.clone()] + maps[set2.clone()] == non_leaf[x][&without_k.to_vec()] {
                let tree1 = reverse_generate_tree(&set1, shortest_paths, maps, graph, non_leaf);
                let tree2 = reverse_generate_tree(&set2, shortest_paths, maps, graph, non_leaf);
                min_edges.extend(&tree1);
                min_edges.extend(&tree2);
                break;
            }
        }
    }

    min_edges
}

/// Calculate minimal Steiner trees of the node sets `T ∪ {v}`
/// where `T ⊆ graph.terminals(), |T| = subset_size` and `v ∈ graph.node_indices() \ T` in which
/// `v` is **not a leaf**.
fn calculate_non_leaf_steiner_trees(
    graph: &Graph,
    maps: &IndexSetMaps<NaturalOrInfinite>,
    sk: &mut Vec<HashMap<Vec<NodeIndex>, NaturalOrInfinite>>,
    subset_size: usize,
) {
    for subset in combinations(graph.terminals(), subset_size) {
        for v in nodes_not_in_subset(graph, &subset) {
            sk[v].insert(
                subset.clone(),
                non_trivial_subsets(&subset)
                    .map(|sub_subset| {
                        let sub_subset_and_v = extend_index(&sub_subset, v);
                        let mut sub_subset_complement_and_v = subset.to_vec();
                        debug_assert!(sorted(&sub_subset));
                        sub_subset_complement_and_v
                            .retain(|e| sub_subset.binary_search(e).is_err());
                        sub_subset_complement_and_v.push(v);
                        maps[sub_subset_and_v] + maps[sub_subset_complement_and_v]
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
fn extend_index<T: Clone>(old_index: &[T], element: T) -> Vec<T> {
    let mut v = old_index.to_vec();
    v.push(element);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::tests::{non_trivial_steiner, small_test_graph};
    use crate::util::TestResult;

    #[test]
    fn test_dreyfus_wagner_trivial() -> TestResult {
        let trivial = small_test_graph()?;
        let result = dreyfus_wagner(&trivial);
        assert_eq!(result, 3.into());
        Ok(())
    }

    #[test]
    fn test_dreyfus_wagner() -> TestResult {
        let graph = non_trivial_steiner()?;
        let result = dreyfus_wagner(&graph);
        assert_eq!(
            result,
            NaturalOrInfinite::from(25 + 30 + 15 + 10 + 40 + 50 + 20)
        );
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
}
