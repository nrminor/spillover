//! Sort whole read pairs with a small spill limit.
//!
//! Run with `cargo run -p spillover-bio --example 08_sort_read_pairs`.

use std::error::Error;

use spillover_bio::{Builder, ReadPair, SeqRecord};

fn main() -> Result<(), Box<dyn Error>> {
    let inputs = [
        (("read-b/1", "TT", "!!"), ("read-b/2", "AC", "##")),
        (("read-a/1", "AA", "!!"), ("read-a/2", "GT", "##")),
    ];
    let mut sorter = Builder::new()
        .sort_by_illumina_pairs()
        .max_buffer_items(1)
        .build();
    for ((name1, sequence1, quality1), (name2, sequence2, quality2)) in inputs {
        let r1 = SeqRecord::new(name1, sequence1, quality1);
        let r2 = SeqRecord::new(name2, sequence2, quality2);
        let pair = ReadPair::from_mates(r1, r2);
        sorter.push(pair)?;
    }
    let output = sorter.finish()?;
    for result in output {
        let pair = result?;
        let (r1, r2) = pair.into_mates();
        println!(
            "{}\t{}",
            String::from_utf8_lossy(r1.name()),
            String::from_utf8_lossy(r2.name())
        );
    }
    Ok(())
}
