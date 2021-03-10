use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::{env, fs};
use steiner_tree::{
    dreyfus_wagner, kou_et_al_steiner_approximation, takahashi_matsuyama_steiner_approximation,
    EdgeTree, Graph,
};

type NodeIndex = usize;

pub fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 1 {
        eprintln!("expected a file name");
        std::process::exit(1);
    }
    let filename = &args[0];
    println!("reading graph...");
    let content = fs::read_to_string(filename)?;
    let graph: Graph = content.parse()?;

    println!("Dreyfus-Wagner algorithm...");
    let tree = dreyfus_wagner(&graph);
    assert_plausible_steiner_tree(&tree, &graph);
    let leaves = tree.find_leaves().len();
    println!("Dreyfus-Wagner tree edges = {}", sorted_edges(&tree));
    println!("Dreyfus-Wagner tree weight = {:?}", tree.weight_in(&graph));
    println!("Writing Dreyfus-Wagner result to file...");
    let mut exact_file = File::create(format!("{}.exact.ost", filename))?;
    tree.write(&mut exact_file, &graph)?;
    println!();

    println!("Kou et al. approximation...");
    let kou_approx = kou_et_al_steiner_approximation(&graph);
    assert_plausible_steiner_tree(&kou_approx, &graph);
    println!(
        "Kou et al. approx. tree edges = {}",
        sorted_edges(&kou_approx)
    );
    println!(
        "Kou et al. approx. tree weight = {:?}",
        kou_approx.weight_in(&graph)
    );
    println!("Writing Kou et al. result to file...");
    let mut kou_file = File::create(format!("{}.kou.ost", filename))?;
    kou_approx.write(&mut kou_file, &graph)?;
    println!();

    println!("Takahashi et al. approximation...");
    let takahashi_approx = takahashi_matsuyama_steiner_approximation(&graph);
    assert_plausible_steiner_tree(&takahashi_approx, &graph);
    println!(
        "Takahashi et al. approx. = {}",
        sorted_edges(&takahashi_approx)
    );
    println!(
        "Takahashi et al. weight = {:?}",
        takahashi_approx.weight_in(&graph)
    );
    println!("Writing Takahashi et al. result to file...");
    let mut takahashi_file = File::create(format!("{}.takahashi.ost", filename))?;
    takahashi_approx.write(&mut takahashi_file, &graph)?;
    println!();

    println!("Checking upper bound...");
    assert!(
        (kou_approx.weight_in(&graph).finite_value() as f64)
            / (tree.weight_in(&graph).finite_value() as f64)
            <= 2.0 - (2.0 / (leaves as f64))
    );
    assert!(
        (takahashi_approx.weight_in(&graph).finite_value() as f64)
            / (tree.weight_in(&graph).finite_value() as f64)
            <= 2.0 - (2.0 / (graph.num_terminals() as f64))
    );
    Ok(())
}

fn sorted_edges(tree: &EdgeTree) -> String {
    let mut edges: Vec<_> = tree.edges().iter().collect();
    edges.sort_unstable();
    format!("{:?}", edges)
}

// Perform some sanity checks on the generated tree.
fn assert_plausible_steiner_tree(tree: &EdgeTree, graph: &Graph) {
    assert_contains_terminals(tree, graph.terminals());
    assert_leaves_are_terminals(tree, graph.terminals());
    assert_tree(tree);
}

// Check that the tree contains all terminals.
fn assert_contains_terminals(tree: &EdgeTree, terminals: &[NodeIndex]) {
    let nodes = tree.nodes();
    assert!(
        terminals.iter().all(|t| nodes.contains(t)),
        "tree {:?} does not contain all terminals ({:?})",
        tree,
        terminals
    );
}

// Check that there are no non-terminal leaves.
fn assert_leaves_are_terminals(tree: &EdgeTree, terminals: &[NodeIndex]) {
    let leaves = tree.find_leaves();
    for leaf in leaves {
        assert!(terminals.contains(&leaf));
    }
}

// Check that the tree is in fact a tree.
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
