extern crate stone_rust;

use clap::{App, Arg};

fn main() {
    let matches = App::new("stonec")
        .arg(Arg::with_name("SRC").index(1).required(true))
        .get_matches();
    if let Err(e) = stone_rust::entry(matches.value_of("SRC").unwrap()) {
        eprintln!("{:?}", e);
    }
}
