#![allow(non_snake_case)] // The non-snake-case name was given on the exam sheet.

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::{env, fs};
use steiner_tree::{dreyfus_wagner, Graph};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let content = fs::read_to_string(&args[0])?;
    let graph: Graph = content.parse()?;
    let tree = dreyfus_wagner(&graph);
    println!("{:?}", tree);
    println!("{:?}", tree.weight_in(&graph));
    Ok(())
}
