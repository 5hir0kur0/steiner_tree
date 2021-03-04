#![allow(non_snake_case)] // The non-snake-case name was given on the exam sheet.

use std::error::Error;
use std::{env, fs};
use steiner_tree::{
    dreyfus_wagner, kou_et_al_steiner_approximation, takahashi_matsuyama_steiner_approximation,
    Graph,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let content = fs::read_to_string(&args[0])?;
    let graph: Graph = content.parse()?;
    let tree = dreyfus_wagner(&graph);
    println!("tree={:?}", tree);
    println!("weight={:?}", tree.weight_in(&graph));
    let kou_approx = kou_et_al_steiner_approximation(&graph);
    println!("kou approx={:?}", kou_approx);
    println!("kou weight={:?}", kou_approx.weight_in(&graph));
    let leaves = tree.find_leaves().len();
    let takahashi_approx = takahashi_matsuyama_steiner_approximation(&graph);
    println!("takahashi approx={:?}", takahashi_approx);
    println!("takahashi weight={:?}", takahashi_approx.weight_in(&graph));
    debug_assert!(
        (kou_approx.weight_in(&graph).finite_value() as f64)
            / (tree.weight_in(&graph).finite_value() as f64)
            <= (2.0 - (2.0 / (leaves as f64)))
    );
    debug_assert!(
        (takahashi_approx.weight_in(&graph).finite_value() as f64)
            / (tree.weight_in(&graph).finite_value() as f64)
            <= (2.0 - (2.0 / (leaves as f64)))
    );
    Ok(())
}
