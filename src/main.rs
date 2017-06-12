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
use std::f64;

use clap::{Arg, App};
use rand::{Rng, Rand};
use rand::distributions::{IndependentSample, Normal, Exp};
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
             .default_value("2048"))
        .arg(Arg::with_name("number")
             .short("n")
             .help("Number of stars, in thousands")
             .takes_value(true)
             .default_value("500"))
        .get_matches();
    let res = args.value_of("resolution").unwrap().parse().chain_err(|| "failed to parse resolution")?;
    let number: usize = args.value_of("number").unwrap().parse().chain_err(|| "failed to parse number of stars")?;
    let number = 1000 * number;
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
    let mut rng = rand::weak_rng();
    let galaxy = Galaxy::rand(&mut rng);
    let viewer = galaxy.star(&mut rng).position;
    let mut max = 0.0;
    // Kahan summation variables
    let mut sum = 0.0;
    let mut c = 0.0;
    for _ in 0..number {
        let star = galaxy.star(&mut rng);
        let vector = star.position - viewer;
        let (face, pos) = project(res, vector);
        let index = address(res, face, pos);
        let out = &mut pixel_data[index as usize];
        let old_irradiance: f32 = out.0.into();
        let old_temp: f32 = out.1.into();

        const SOLAR_LUMINOSITY: f64 = 3.828e26;
        const GALAXY_RADIUS: f64 = 1e21;
        // Conversion from solar luminances per galaxy radius^2 to attowatts/m^2
        const SCALING_FACTOR: f32 = (1e15 * (SOLAR_LUMINOSITY / (GALAXY_RADIUS * GALAXY_RADIUS))) as f32;

        let irradiance = SCALING_FACTOR * star.intensity / na::norm(&vector).powi(2);
        if irradiance > max { max = irradiance; }
        if old_irradiance + irradiance > 0.0 {
            *out = (f16::from_f32((old_irradiance + irradiance).min(half::consts::MAX.into())),
                    f16::from_f32((old_temp * old_irradiance + star.temperature * irradiance)
                                  / (old_irradiance + irradiance)));
        }

        // Kahan summation
        let y = irradiance - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    println!("brightest star's irradiance: {} fW/m^2\ntotal irradiance: {} fW/m^2", max, sum);

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
        let y = Normal::new(0.0, 0.25);
        let xz = Normal::new(0.0, 1.0);
        let pos = na::Point3::new(xz.ind_sample(rng) as f32, y.ind_sample(rng) as f32, xz.ind_sample(rng) as f32);
        let pos = self.rotation * pos;

        //
        // units below are wrt. sol
        //

        let mass = Exp::new(1.0).ind_sample(rng);

        let radius = 0.43039846 * mass + 0.52963256; // TODO: Fudge

        // Mass-luminosity relation
        // Main-Sequence Effective Temperatures from a Revised Mass-Luminosity Relation Based on Accurate Properties
        // Z. Eker, F. Soydugan, E. Soydugan, S. Bilir, E. Yaz Gokce, I. Steer, M. Tuysuz, T. Senyuz, O. Demircan (2015)
        let luminosity = if mass <= 1.05 {
            4.841132 * mass.ln() - 0.02625
        } else if mass <= 2.40 {
            4.32891 * mass.ln() - 0.00220
        } else if mass <= 7.0 {
            3.962203 * mass.ln() + 0.120112
        } else {
            2.726203 * mass.ln() + 1.237228
        }.exp();

        let temperature = 5777.0 * (luminosity / radius.powi(2)).powf(0.25);

        Star {
            position: pos,
            intensity: (luminosity / (4.0 * f64::consts::PI)) as f32,
            temperature: temperature as f32,
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
    /// Radiant intensity
    intensity: f32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Face {
    PX = 0,
    NX = 1,
    PY = 2,
    NY = 3,
    PZ = 4,
    NZ = 5,
}

fn project(res: u32, n: na::Vector3<f32>) -> (Face, na::Vector2<f32>) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let face;
    let pos;
    if ax >= ay && ax >= az {
        if ax == 0.0 {
            face = Face::PX;
            pos = na::zero();
        } else {
            pos = na::Vector2::new((n.y / ax + 1.0) / 2.0 * (res - 1) as f32,
                                   (n.z / ax + 1.0) / 2.0 * (res - 1) as f32);
            face = if n.x > 0.0 {
                Face::PX
            } else {
                Face::NX
            };
        }
    } else if ay >= az {
        pos = na::Vector2::new((n.x / ay + 1.0) / 2.0 * (res - 1) as f32,
                               (n.z / ay + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.y > 0.0 {
            Face::PY
        } else {
            Face::NY
        };
    } else {
        pos = na::Vector2::new((n.x / az + 1.0) / 2.0 * (res - 1) as f32,
                               (n.y / az + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.z > 0.0 {
            Face::PZ
        } else {
            Face::NZ
        };
    }
    (face, pos)
}

fn address(res: u32, face: Face, pos: na::Vector2<f32>) -> u32 {
    let y_min = (face as u32 * res) as f32;
    let y_max = y_min + (res - 1) as f32;
    let x_max = (res - 1) as f32;
    let x;
    let y;
    match face {
        Face::PX => {
            x = pos.y;
            y = y_max - pos.x;
        }
        Face::NX => {
            x = x_max - pos.y;
            y = y_max - pos.x;
        }
        Face::PY => {
            x = pos.x;
            y = y_max - pos.y;
        }
        Face::NY => {
            x = pos.x;
            y = y_min + pos.y;
        }
        Face::PZ => {
            x = x_max - pos.x;
            y = y_max - pos.y;
        }
        Face::NZ => {
            x = pos.x;
            y = y_max - pos.y;
        }
    }
    x as u32 + y as u32 * res
}

#[test]
fn project_sanity() {
    assert_eq!(project(128, na::Vector3::x()), (Face::PX, na::Vector2::new(63.5, 63.5)));
    assert_eq!(project(128, na::Vector3::y()), (Face::PY, na::Vector2::new(63.5, 63.5)));
    assert_eq!(project(128, na::Vector3::z()), (Face::PZ, na::Vector2::new(63.5, 63.5)));
    assert_eq!(project(128, -na::Vector3::x()), (Face::NX, na::Vector2::new(63.5, 63.5)));
    assert_eq!(project(128, -na::Vector3::y()), (Face::NY, na::Vector2::new(63.5, 63.5)));
    assert_eq!(project(128, -na::Vector3::z()), (Face::NZ, na::Vector2::new(63.5, 63.5)));
}
