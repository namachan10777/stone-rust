extern crate stone_rust;

use clap::{App, Arg};

fn main() {
    let matches = App::new("stonec")
        .arg(Arg::with_name("SRC").index(1).required(true))
        .get_matches();
    match stone_rust::entry(matches.value_of("SRC").unwrap()) {
        Ok(s) => println!("{}", s),
        Err(e) => eprintln!("{:?}", e),
    }
}
