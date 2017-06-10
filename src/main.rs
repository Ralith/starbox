extern crate openexr as exr;
extern crate clap;
#[macro_use]
extern crate error_chain;
extern crate nalgebra as na;
extern crate rand;
extern crate half;

error_chain! {}

use std::fs::File;
use std::path::Path;

use clap::{Arg, App};
use rand::{Rng, Rand};
use rand::distributions::{IndependentSample, Range};
use half::f16;

quick_main!(run);

fn run() -> Result<()> {
    let args = App::new("starbox")
        .version("0.1")
        .author("Benjamin Saunders <ben.e.saunders@gmail.com>")
        .about("Generates starboxes")
        .arg(Arg::with_name("FILE")
             .help("File to write")
             .required(true))
        .arg(Arg::with_name("resolution")
             .short("r")
             .help("Cubemap edge length")
             .takes_value(true)
             .default_value("1024"))
        .get_matches();
    let res = args.value_of("resolution").unwrap().parse().chain_err(|| "failed to parse resolution")?;
    let path = Path::new(args.value_of_os("FILE").unwrap());
    let mut out = File::create(path).chain_err(|| "failed to open output file")?;
    let mut out = exr::ScanlineOutputFile::new(
        &mut out,
        exr::Header::new()
            .set_resolution(res, 6*res)
            .set_envmap(Some(exr::Envmap::Cube))
            .add_channel("R", exr::PixelType::HALF)
            .add_channel("G", exr::PixelType::HALF)
            .add_channel("B", exr::PixelType::HALF))
        .chain_err(|| "failed to initialize encoder")?;

    let zero = f16::from_f32(0.0);
    let mut pixel_data: Vec<(f16, f16, f16)> = vec![(zero, zero, zero); (res * 6 * res) as usize];
    const COUNT: usize = 20_000;
    let mut rng = rand::weak_rng();
    for _ in 0..COUNT {
        let star = Star::rand(&mut rng);
        let (face, pos) = project(res, star.direction);
        pixel_data[(pos.0 + res * (pos.1 + res * face as u32)) as usize] = star.irradiance;
    }

    {
        let mut fb = exr::FrameBuffer::new(res, 6*res);
        fb.insert_channels(&["R", "G", "B"], &pixel_data);
        out.write_pixels(&fb).chain_err(|| "failed to output data")?;
    }

    Ok(())
}

struct Star {
    direction: na::Unit<na::Vector3<f32>>,
    irradiance: (f16, f16, f16),
}

impl Rand for Star {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        let one = f16::from_f32(1.0);
        Star {
            direction: dir(rng),
            irradiance: (one, one, one),
        }
    }
}

// Choosing a Point from the Surface of a Sphere. George Marsaglia (2007)
fn dir<R: Rng>(rng: &mut R) -> na::Unit<na::Vector3<f32>> {
    let between = Range::new(-1.0, 1.0); // FIXME: Should be open, not half-open
    let mut v1: f32 = between.ind_sample(rng);
    let mut v2: f32;
    let mut s;
    loop {
        v2 = between.ind_sample(rng);
        s = v1.powi(2) + v2.powi(2);
        if s < 1.0 { break; }
        v1 = v2;
    }
    let a = (1.0 - s).sqrt();
    na::Unit::new_unchecked(na::Vector3::new(2.0 * v1 * a, 2.0 * v2 * a, 1.0 - 2.0 * s))
}

enum Face {
    PX = 0,
    NX = 1,
    PY = 2,
    NY = 3,
    PZ = 4,
    NZ = 5,
}

fn project(res: u32, n: na::Unit<na::Vector3<f32>>) -> (Face, (u32, u32)) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let face: Face;
    let pos: (f32, f32);
    if ax >= ay && ax >= az {
        pos = ((n.y / ax + 1.0) / 2.0 * (res - 1) as f32,
               (n.z / ax + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.x > 0.0 {
            Face::PX
        } else {
            Face::NX
        };
    } else if ay >= az {
        pos = ((n.x / ay + 1.0) / 2.0 * (res - 1) as f32,
               (n.z / ay + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.y > 0.0 {
            Face::PY
        } else {
            Face::NY
        };
    } else {
        pos = ((n.x / az + 1.0) / 2.0 * (res - 1) as f32,
               (n.y / az + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.z > 0.0 {
            Face::PZ
        } else {
            Face::NZ
        };
    }
    (face, (pos.0 as u32, pos.1 as u32))
}
