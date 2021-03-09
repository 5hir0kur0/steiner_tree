#![allow(non_snake_case)] // The non-snake-case name was given on the exam sheet.

//! Benchmark a given graph.
//! Output the result to stdout and (error) messages to stderr.

use std::error::Error;
use std::fs::File;
use std::time::{Duration, Instant};
use std::{env, fs, io};
use steiner_tree::{
    dreyfus_wagner, kou_et_al_steiner_approximation, takahashi_matsuyama_steiner_approximation,
    takahashi_matsuyama_steiner_approximation_serial, EdgeTree, Graph,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && &args[0] == "header" {
        println!("# delete this comment and always join 5 lines of text to get the final CSV");
        println!(
            "filename, num_edges, num_nodes, num_terminals, average_degree, min_degree, \
                      max_degree, average_weight, min_weight, max_weight, "
        );
        println!("dreyfus_weight, dreyfus_time, ");
        println!("kou_weight, kou_time, ");
        println!("takahashi_weight_serial, takahashi_time_serial, ");
        println!("takahashi_weight_parallel, takahashi_time_parallel");
        return Ok(());
    }
    if args.len() != 2 {
        eprintln!("usage: {{header, stats, exact, approx1, approx2s, approx2p}} <FILENAME>");
        eprintln!("  Arguments should always be used in this order on one file");
        eprintln!("  and stdout should be appended to a file to get the CSV data.");
        eprintln!("  See benchmark.sh script in this repo.");
        std::process::exit(1);
    }
    let filename = &args[1];
    eprintln!("reading graph...");
    let content = fs::read_to_string(filename)?;
    let graph: Graph = content.parse()?;
    match &args[0] as &str {
        "stats" => {
            let stats = GraphStatistics::new(&graph);
            println!(
                "{fp}, {ne}, {nn}, {nt}, {ad}, {md}, {Md}, {aw}, {mw}, {Mw}, ",
                fp = filename,
                ne = stats.num_edges,
                nn = stats.num_nodes,
                nt = stats.num_terminals,
                ad = stats.average_degree,
                md = stats.min_degree,
                Md = stats.max_degree,
                aw = stats.average_weight,
                mw = stats.min_weight,
                Mw = stats.max_weight,
            );
        }
        "exact" => {
            benchmark_alg("dreyfus-wagner", dreyfus_wagner, filename, &graph)?;
        }
        "approx1" => {
            benchmark_alg("kou", kou_et_al_steiner_approximation, filename, &graph)?;
        }
        "approx2s" => {
            benchmark_alg(
                "takahashi-serial",
                takahashi_matsuyama_steiner_approximation_serial,
                filename,
                &graph,
            )?;
        }
        "approx2p" => {
            benchmark_alg(
                "takahashi-parallel",
                takahashi_matsuyama_steiner_approximation,
                filename,
                &graph,
            )?;
        }
        arg => {
            eprintln!("invalid argument: {}", arg);
            eprintln!("expected one of: header, stats, exact, approx1, approx2s, approx2p");
            std::process::exit(1);
        }
    }
    Ok(())
}

struct GraphStatistics {
    num_edges: usize,
    num_nodes: usize,
    num_terminals: usize,
    average_degree: f64,
    min_degree: usize,
    max_degree: usize,
    average_weight: f64,
    min_weight: u32,
    max_weight: u32,
}

impl GraphStatistics {
    pub fn new(graph: &Graph) -> Self {
        let num_terminals = graph.num_terminals();
        let num_nodes = graph.num_nodes();
        let num_edges = graph.edges().count();
        let degrees = graph
            .node_indices()
            .map(|ni| graph.neighbors(ni).count())
            .collect::<Vec<_>>();
        let average_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
        let max_degree = degrees.iter().max().copied().unwrap();
        let min_degree = degrees.iter().min().copied().unwrap();
        let weights = graph
            .edges()
            .map(|(_, _, weight)| weight)
            .collect::<Vec<_>>();
        let min_weight = weights.iter().min().copied().unwrap();
        let max_weight = weights.iter().max().copied().unwrap();
        let average_weight = weights.iter().sum::<u32>() as f64 / weights.len() as f64;
        Self {
            num_edges,
            num_nodes,
            num_terminals,
            average_degree,
            min_degree,
            max_degree,
            average_weight,
            min_weight,
            max_weight,
        }
    }
}

// benchmark an algorithm, output running time and tree weight
fn benchmark_alg<F: Fn(&Graph) -> EdgeTree, P: AsRef<str>>(
    name: &str,
    alg: F,
    filename: P,
    graph: &Graph,
) -> io::Result<()> {
    eprintln!("running {}...", name);
    let (tree, time) = measure_time(|| alg(&graph));
    eprintln!("{} tree = {:?}", name, tree);
    let weight = tree.weight_in(&graph);
    eprintln!("{} weight = {:?}", name, weight);
    println!("{}, {}, ", weight.finite_value(), time.as_nanos());
    let mut exact_file = File::create(format!("{}.exact.ost", filename.as_ref()))?;
    tree.write(&mut exact_file, &graph)
}

// measure running time of a closure
fn measure_time<F: FnOnce() -> R, R>(closure: F) -> (R, Duration) {
    let before = Instant::now();
    let result = closure();
    (result, before.elapsed())
}
