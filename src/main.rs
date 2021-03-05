#![allow(non_snake_case)] // The non-snake-case name was given on the exam sheet.

use std::error::Error;
use std::{env, fs};
use steiner_tree::{
    dreyfus_wagner, kou_et_al_steiner_approximation, takahashi_matsuyama_steiner_approximation,
    Graph,
};
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let filename = &args[0];
    let content = fs::read_to_string(filename)?;
    let graph: Graph = content.parse()?;
    let tree = dreyfus_wagner(&graph);
    println!("tree={:?}", tree);
    println!("weight={:?}", tree.weight_in(&graph));
    let mut exact_file = File::create(format!("{}.exact.ost", filename))?;
    tree.write(&mut exact_file, &graph)?;
    let kou_approx = kou_et_al_steiner_approximation(&graph);
    println!("kou approx={:?}", kou_approx);
    println!("kou weight={:?}", kou_approx.weight_in(&graph));
    let mut kou_file = File::create(format!("{}.kou.ost", filename))?;
    kou_approx.write(&mut kou_file, &graph)?;
    let leaves = tree.find_leaves().len();
    let takahashi_approx = takahashi_matsuyama_steiner_approximation(&graph);
    println!("takahashi approx={:?}", takahashi_approx);
    println!("takahashi weight={:?}", takahashi_approx.weight_in(&graph));
    let mut takahashi_file = File::create(format!("{}.takahashi.ost", filename))?;
    takahashi_approx.write(&mut takahashi_file, &graph)?;
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
