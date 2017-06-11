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
use rand::distributions::{IndependentSample, Normal};
use rand::distributions::normal::StandardNormal;
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
    println!("uncompressed size: {} MiB", res * res * 6 * 2 * 2 / (1024 * 1024));
    let path = Path::new(args.value_of_os("FILE").unwrap());
    let mut out = File::create(path).chain_err(|| "failed to open output file")?;
    let mut out = exr::ScanlineOutputFile::new(
        &mut out,
        exr::Header::new()
            .set_resolution(res, 6*res)
            .set_envmap(Some(exr::Envmap::Cube))
            .add_channel("Y", exr::PixelType::HALF)
            .add_channel("T", exr::PixelType::HALF))
        .chain_err(|| "failed to initialize encoder")?;

    let zero = f16::from_f32(0.0);
    let mut pixel_data: Vec<(f16, f16)> = vec![(zero, zero); (res * 6 * res) as usize];
    const COUNT: usize = 40_000;
    let mut rng = rand::weak_rng();
    let galaxy = Galaxy::rand(&mut rng);
    let viewer = galaxy.star(&mut rng).position;
    for _ in 0..COUNT {
        let star = galaxy.star(&mut rng);
        let (face, pos) = project(res, star.position - viewer);
        let out = &mut pixel_data[(pos.0 + res * (pos.1 + res * face as u32)) as usize];
        let old_irradiance: f32 = out.0.into();
        let old_temp: f32 = out.1.into();
        *out = (f16::from_f32(old_irradiance + star.irradiance),
                f16::from_f32(old_temp * old_irradiance + star.temperature * star.irradiance
                              / (old_irradiance + star.irradiance)));
    }

    {
        let mut fb = exr::FrameBuffer::new(res, 6*res);
        fb.insert_channels(&["Y", "T"], &pixel_data);
        out.write_pixels(&fb).chain_err(|| "failed to output data")?;
    }

    Ok(())
}

struct Galaxy {
    rotation: na::UnitQuaternion<f32>,
}

impl Galaxy {
    fn star<R: Rng>(&self, rng: &mut R) -> Star {
        let y = Normal::new(0.0, 1.0);
        let xz = Normal::new(0.0, 4.0);
        let pos = na::Point3::new(xz.ind_sample(rng) as f32, y.ind_sample(rng) as f32, xz.ind_sample(rng) as f32);
        let pos = self.rotation * pos;

        Star {
            position: pos,
            irradiance: 1.0,
            temperature: 1.0,
        }
    }
}

impl Rand for Galaxy {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        let StandardNormal(x) = rng.gen();
        let StandardNormal(y) = rng.gen();
        let StandardNormal(z) = rng.gen();
        let StandardNormal(w) = rng.gen();
        let q = na::Unit::new_normalize(na::Quaternion::new(w, x, y, z));
        Galaxy {
            rotation: na::convert(q),
        }
    }
}

struct Star {
    position: na::Point3<f32>,
    temperature: f32,
    irradiance: f32,
}

enum Face {
    PX = 0,
    NX = 1,
    PY = 2,
    NY = 3,
    PZ = 4,
    NZ = 5,
}

fn project(res: u32, n: na::Vector3<f32>) -> (Face, (u32, u32)) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let face: Face;
    let pos: (f32, f32);
    if ax >= ay && ax >= az {
        if ax == 0.0 {
            return (Face::PX, (0, 0));
        }
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
