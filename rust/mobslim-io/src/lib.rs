//! MATSim population XML (`plans_v4`/`population_v6`) read straight into
//! the columns `mobslim.formats.matsim_plans.read_population_xml` builds.
//!
//! This is a fast path for that Python reader, not a second definition of
//! the format: it reads what the Python reader reads to exactly the same
//! values, and raises `FallBack` for everything else (any error, and
//! inputs it does not handle: `plans_v4` node routes with `keep_routes`,
//! encodings other than UTF-8, namespaces, DTD entities, ...). The Python
//! reader then reads the file and gives its result or its error.
//!
//! One thread decompresses the file and cuts it into chunks before a
//! `<person` (`split`); worker threads parse the chunks (`parse_chunk`) and
//! their columns are joined in file order (`merge`). A cut that is not
//! before a person of the document element (one in a comment, say) leaves a
//! chunk that does not parse as whole elements, so the file falls back.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::sync::mpsc;
use std::sync::Mutex;

use flate2::read::MultiGzDecoder;
use numpy::IntoPyArray;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use memchr::memmem;
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

create_exception!(_kernel, FallBack, PyException, "Read this file with the Python reader.");

/// `Err`: fall back to the Python reader (the reason is for debugging).
type R<T> = Result<T, String>;

macro_rules! bail {
    ($($t:tt)*) => { return Err(format!($($t)*)) };
}

// ------------------------------------------------------------ values

fn utf8(b: &[u8]) -> R<&str> {
    std::str::from_utf8(b).map_err(|_| "not UTF-8".to_string())
}

fn unescape(s: Cow<'_, str>) -> R<Cow<'_, str>> {
    if !s.contains('&') {
        return Ok(s);
    }
    match quick_xml::escape::unescape(&s) {
        Ok(u) => Ok(Cow::Owned(u.into_owned())),
        Err(e) => bail!("entity: {e}"),
    }
}

/// An attribute value as expat gives it: literal tabs and line ends become
/// spaces (CDATA attribute normalisation), then references are expanded.
fn attr_value(raw: &[u8]) -> R<Cow<'_, str>> {
    let s = utf8(raw)?;
    if s.bytes().any(|c| matches!(c, b'\t' | b'\n' | b'\r')) {
        unescape(Cow::Owned(s.replace("\r\n", " ").replace(['\t', '\n', '\r'], " ")))
    } else {
        unescape(Cow::Borrowed(s))
    }
}

/// Character data as expat gives it: line ends normalised to `\n`.
fn newlines(s: &str) -> Cow<'_, str> {
    if s.contains('\r') {
        Cow::Owned(s.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(s)
    }
}

/// Python's `float(s)` for the strings Rust parses (a subset of Python's,
/// to the same value); anything else falls back.
fn pyfloat(s: &str) -> R<f64> {
    s.parse::<f64>().map_err(|_| format!("float {s:?}"))
}

/// Python's `int(s)` for plain ASCII digits.
fn pyint(s: &str) -> R<i64> {
    if s.is_empty() || s.len() > 15 || !s.bytes().all(|c| c.is_ascii_digit()) {
        bail!("int {s:?}");
    }
    Ok(s.parse::<i64>().unwrap())
}

/// Python's `int(s)` for an optional sign and ASCII digits.
fn pyint_signed(s: &str) -> R<i64> {
    match s.as_bytes().first() {
        Some(b'-') => Ok(-pyint(&s[1..])?),
        Some(b'+') => pyint(&s[1..]),
        _ => pyint(s),
    }
}

/// A typed MATSim attribute value.
enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

/// `mobslim.formats.matsim_attributes.parse_matsim_attribute`.
fn attribute_value(class: Option<&str>, text: String) -> R<Value> {
    Ok(match class {
        Some("java.lang.Integer" | "java.lang.Long" | "java.lang.Short" | "java.lang.Byte") => {
            Value::Int(pyint_signed(text.trim())?)
        }
        Some("java.lang.Double" | "java.lang.Float") => Value::Float(pyfloat(text.trim())?),
        Some("java.lang.Boolean") => Value::Bool(text.trim().to_lowercase() == "true"),
        _ => Value::Str(text),
    })
}

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(match self {
            Value::Int(i) => i.into_pyobject(py)?.into_any(),
            Value::Float(f) => f.into_pyobject(py)?.into_any(),
            Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
            Value::Str(s) => s.into_pyobject(py)?.into_any(),
        })
    }
}

/// `mobslim.timeutils.parse_time`.
fn parse_time(s: &str) -> R<f64> {
    let parts: Vec<&str> = s.trim().split(':').collect();
    let (h, m, sec) = match parts.as_slice() {
        [h, m] => (*h, *m, "0"),
        [h, m, s] => (*h, *m, *s),
        _ => bail!("time {s:?}"),
    };
    Ok((pyint(h)? * 3600 + pyint(m)? * 60) as f64 + pyfloat(sec)?)
}

/// `numpy.sum` of a float64 array (pairwise summation, as numpy 2 does it).
fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

// ------------------------------------------------------------ network

struct Net {
    link: HashMap<String, i32>,
    node: HashMap<String, i32>,
    to_node: Vec<i64>,
    fstt: Vec<f64>,
    length: Vec<f64>,
}

fn id_map(ids: Vec<Option<String>>) -> HashMap<String, i32> {
    let mut map = HashMap::with_capacity(ids.len());
    for (i, id) in ids.into_iter().enumerate() {
        if let Some(id) = id {
            map.entry(id).or_insert(i as i32);
        }
    }
    map
}

impl Net {
    fn link_index(&self, id: &str) -> R<i32> {
        self.link.get(id).copied().ok_or_else(|| format!("unknown link {id:?}"))
    }
}

// ------------------------------------------------------------ elements

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Root,
    Person,
    PersonAttrs,
    PersonAttr,
    Plan,
    Act,
    Leg,
    LegAttrs,
    LegAttr,
    Route,
    Other,
}

const V4_KEYS: [&str; 5] = ["sex", "age", "license", "car_avail", "employed"];

type Attr<'b> = Option<Cow<'b, str>>;

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// Calls `f(name, raw value)` for each attribute of a start tag (`raw`: what
/// follows the tag name), checking what expat checks: `name="value"` pairs
/// (either quote) separated by spaces, no `<` in values.
fn for_each_attr<'b>(raw: &'b [u8], mut f: impl FnMut(&'b [u8], &'b [u8]) -> R<()>) -> R<()> {
    let n = raw.len();
    let mut i = 0;
    loop {
        let before = i;
        while i < n && is_space(raw[i]) {
            i += 1;
        }
        if i == n {
            return Ok(());
        }
        if i == before {
            bail!("attributes not separated by space");
        }
        let k0 = i;
        while i < n && !is_space(raw[i]) && raw[i] != b'=' {
            i += 1;
        }
        let key = &raw[k0..i];
        while i < n && is_space(raw[i]) {
            i += 1;
        }
        if i == n || raw[i] != b'=' {
            bail!("attribute without value");
        }
        i += 1;
        while i < n && is_space(raw[i]) {
            i += 1;
        }
        if i == n || !matches!(raw[i], b'"' | b'\'') {
            bail!("unquoted attribute value");
        }
        let (quote, v0) = (raw[i], i + 1);
        i = v0;
        while i < n && raw[i] != quote {
            if raw[i] == b'<' {
                bail!("< in attribute value");
            }
            i += 1;
        }
        if i == n {
            bail!("unterminated attribute value");
        }
        f(key, &raw[v0..i])?;
        i += 1;
    }
}

/// A namespace declaration or a prefixed name other than `xml:` (always bound;
/// expat accepts `xml:lang`, which old MATSim files put on the root).
fn namespaced(key: &[u8]) -> bool {
    key.starts_with(b"xmlns") || (memchr::memchr(b':', key).is_some() && !key.starts_with(b"xml:"))
}

/// Checks every attribute of `e` (see `for_each_attr`; also no namespaces
/// and no duplicates) and returns the values of `names` (`None`: absent),
/// borrowed where they need no unescaping. The input is valid UTF-8.
fn read_attrs<'b, const N: usize>(e: &'b BytesStart, names: [&[u8]; N]) -> R<[Attr<'b>; N]> {
    let mut out: [Attr<'b>; N] = std::array::from_fn(|_| None);
    let mut seen: [&[u8]; 32] = [b""; 32];
    let mut n_seen = 0;
    for_each_attr(e.attributes_raw(), |key, value| {
        if key.is_empty() || seen[..n_seen].contains(&key) || n_seen == seen.len() {
            bail!("bad, duplicate or too many attributes");
        }
        seen[n_seen] = key;
        n_seen += 1;
        if namespaced(key) {
            bail!("namespaced attribute");
        }
        if let Some(k) = names.iter().position(|n| *n == key) {
            out[k] = Some(attr_value(value)?);
        }
        Ok(())
    })?;
    Ok(out)
}

fn non_empty<'s>(s: &'s Option<impl AsRef<str>>) -> Option<&'s str> {
    s.as_ref().map(|s| s.as_ref()).filter(|s| !s.is_empty())
}

fn owned(s: Attr<'_>) -> Option<String> {
    s.map(Cow::into_owned)
}

struct RouteInfo {
    typ: Option<String>,
    start: Option<String>,
    end: Option<String>,
    text: String,
}

#[derive(Default)]
struct LegInfo {
    mode: Option<String>,
    trav_time: Option<String>,
    routing: Option<String>, // the first `routingMode` of its first <attributes>
    route: Option<RouteInfo>, // its first <route>
    attrs_seen: bool,
    route_seen: bool,
}

enum TripLeg {
    Mode(String),  // a leg to route
    File(LegInfo), // a single leg, whose route may be kept
}

struct Act {
    typ: i32, // code into Columns::act_types
    node: i32, // -1: the node nearest x, y (resolved in Python)
    end: f64,
    dur: f64,
    x: f64,
    y: f64,
    link: i32,
}

/// `_trip_leg`: one leg for a trip read as `leg (stage leg)*`.
fn trip_leg(trip: Vec<Option<LegInfo>>) -> R<TripLeg> {
    const TRIP_WALKS: [&str; 5] =
        ["walk", "non_network_walk", "transit_walk", "access_walk", "egress_walk"];
    let mut legs: Vec<LegInfo> = trip.into_iter().flatten().collect();
    if legs.is_empty() {
        bail!("trip without legs");
    }
    if legs.len() == 1 && legs[0].mode.as_deref() != Some("transit_walk") {
        let leg = legs.pop().unwrap();
        if let Some(r) = &leg.routing {
            if Some(r.as_str()) != leg.mode.as_deref() {
                return Ok(TripLeg::Mode(r.clone()));
            }
        }
        if leg.mode.is_none() {
            bail!("leg without mode");
        }
        return Ok(TripLeg::File(leg));
    }
    if let Some(r) = legs.iter().find_map(|l| l.routing.clone()) {
        return Ok(TripLeg::Mode(r));
    }
    let main = legs.iter().find(|l| !l.mode.as_deref().is_some_and(|m| TRIP_WALKS.contains(&m)));
    let mode = match main {
        Some(l) => l.mode.clone(),
        None if legs.iter().any(|l| l.mode.as_deref() == Some("transit_walk")) => {
            Some("pt".to_string())
        }
        None => legs[0].mode.clone(),
    };
    mode.map(TripLeg::Mode).ok_or_else(|| "leg without mode".to_string())
}

// ------------------------------------------------------------ output

#[derive(Default)]
struct Columns {
    type_code: HashMap<String, i32>,
    act_types: Vec<String>,
    mode_code: HashMap<String, i32>,
    modes: Vec<String>,
    act_offsets: Vec<i64>,
    a_type: Vec<i32>,
    a_node: Vec<i32>,
    a_end: Vec<f64>,
    a_dur: Vec<f64>,
    a_x: Vec<f64>,
    a_y: Vec<f64>,
    a_link: Vec<i32>,
    l_mode: Vec<i32>,
    l_tt: Vec<f64>,
    l_routed: Vec<bool>,
    l_rtt: Vec<f64>,
    l_rdist: Vec<f64>,
    route_offsets: Vec<i64>,
    route_links: Vec<i32>,
    // persons
    ids: Vec<String>,
    n_plans: Vec<i64>,
    selected: Vec<i64>,
    scores: Vec<f64>,
    // (person, plans_v4 attributes, <attributes> entries (name, value))
    #[allow(clippy::type_complexity)]
    attrs: Vec<(i64, Vec<(&'static str, String)>, Vec<(Option<String>, Value)>)>,
}

impl Columns {
    fn new() -> Self {
        Columns { act_offsets: vec![0], route_offsets: vec![0], ..Default::default() }
    }
}

/// `name`'s code (`PlanBatchBuilder` numbers names in order of first use).
fn code(map: &mut HashMap<String, i32>, names: &mut Vec<String>, name: &str) -> i32 {
    if let Some(&c) = map.get(name) {
        return c;
    }
    let c = names.len() as i32;
    map.insert(name.to_string(), c);
    names.push(name.to_string());
    c
}

impl Columns {
    fn act(&mut self, a: Act) {
        self.a_type.push(a.typ);
        self.a_node.push(a.node);
        self.a_end.push(a.end);
        self.a_dur.push(a.dur);
        self.a_x.push(a.x);
        self.a_y.push(a.y);
        self.a_link.push(a.link);
    }

    /// `PlanBatchBuilder.leg`; `route`: (links, route time, distance).
    fn leg(&mut self, mode: &str, tt: f64, route: Option<(Vec<i32>, f64, f64)>) {
        let c = code(&mut self.mode_code, &mut self.modes, mode);
        self.l_mode.push(c);
        self.l_tt.push(tt);
        match route {
            None => {
                self.l_routed.push(false);
                self.l_rtt.push(f64::NAN);
                self.l_rdist.push(f64::NAN);
            }
            Some((links, rtt, rdist)) => {
                self.l_routed.push(true);
                self.l_rtt.push(rtt);
                self.l_rdist.push(rdist);
                self.route_links.extend(links);
            }
        }
        self.route_offsets.push(self.route_links.len() as i64);
    }

    /// A single-leg trip: `_file_route` if `keep_routes`, else unrouted.
    fn file_leg(&mut self, leg: &LegInfo, net: &Net, keep_routes: bool) -> R<()> {
        let mode = leg.mode.as_deref().unwrap();
        let tt = match non_empty(&leg.trav_time) {
            Some(t) => parse_time(t)?,
            None => f64::NAN,
        };
        let route = match (&leg.route, keep_routes) {
            (Some(r), true) => r,
            _ => {
                self.leg(mode, tt, None);
                return Ok(());
            }
        };
        if route.start.is_none() && route.typ.is_none() {
            bail!("plans_v4 node route");
        }
        if route.typ.as_deref().unwrap_or("links") != "links" || route.start.is_none() {
            self.leg(mode, tt, None); // a generic/pt route: route it
            return Ok(());
        }
        let ids: Vec<&str> = route.text.split_whitespace().collect();
        if ids.is_empty()
            || Some(ids[0]) != route.start.as_deref()
            || Some(ids[ids.len() - 1]) != route.end.as_deref()
        {
            bail!("route endpoints");
        }
        let links = ids[1..].iter().map(|i| net.link_index(i)).collect::<R<Vec<i32>>>()?;
        let (rtt, rdist) = if links.is_empty() {
            (0.0, 0.0)
        } else {
            let tt: Vec<f64> = links.iter().map(|&k| net.fstt[k as usize]).collect();
            let len: Vec<f64> = links.iter().map(|&k| net.length[k as usize]).collect();
            (np_sum(&tt), np_sum(&len))
        };
        self.leg(mode, tt, Some((links, rtt, rdist)));
        Ok(())
    }
}

// ------------------------------------------------------------ reading

#[derive(Default)]
struct PlanState {
    acts: Vec<Act>,
    legs: Vec<TripLeg>,
    trip: Vec<Option<LegInfo>>, // legs (None: stage activities) of the current trip
    score: f64,
}

#[derive(Default)]
struct PersonState {
    n_plans: i64,
    selected: i64,
    attrs_seen: bool,
    v4: Vec<(&'static str, String)>,
    entries: Vec<(Option<String>, Value)>,
}

struct Text {
    depth: usize,
    open: bool, // no child element yet: ElementTree's `.text`
    buf: String,
}

struct Parser<'a> {
    net: &'a Net,
    keep_routes: bool,
    out: Columns,
    stack: Vec<Kind>, // from the document element (read by `split`)
    person: PersonState,
    plan: PlanState,
    leg: LegInfo,
    attr: (Option<String>, Option<String>), // the person <attribute> being read
    text: Option<Text>,
}

impl<'a> Parser<'a> {
    fn start(&mut self, e: &BytesStart) -> R<()> {
        let name = e.name();
        let name = name.as_ref();
        if name.contains(&b':') {
            bail!("namespaced element");
        }
        if let Some(t) = &mut self.text {
            t.open = false;
        }
        let parent = self.stack.last().copied();
        let kind = match (parent, name) {
            (None, _) => bail!("unbalanced chunk"),
            (Some(Kind::Root), b"person") => Kind::Person,
            (Some(Kind::Person), b"attributes") if !self.person.attrs_seen => Kind::PersonAttrs,
            (Some(Kind::PersonAttrs), b"attribute") => Kind::PersonAttr,
            (Some(Kind::Person), b"plan") => Kind::Plan,
            (Some(Kind::Plan), b"act" | b"activity") => Kind::Act,
            (Some(Kind::Plan), b"leg") => Kind::Leg,
            (Some(Kind::Leg), b"attributes") if !self.leg.attrs_seen => Kind::LegAttrs,
            (Some(Kind::LegAttrs), _) => Kind::LegAttr,
            (Some(Kind::Leg), b"route") if !self.leg.route_seen => Kind::Route,
            _ => Kind::Other,
        };
        match kind {
            Kind::Person => {
                let [id, sex, age, license, car_avail, employed] = read_attrs(
                    e, [b"id", b"sex", b"age", b"license", b"car_avail", b"employed"])?;
                self.out.ids.push(owned(id).ok_or("person without id")?);
                self.person = PersonState::default();
                for (key, val) in V4_KEYS.iter().zip([sex, age, license, car_avail, employed]) {
                    if let Some(val) = val {
                        self.person.v4.push((key, val.into_owned()));
                    }
                }
            }
            Kind::PersonAttrs => {
                read_attrs(e, [])?;
                self.person.attrs_seen = true;
            }
            Kind::PersonAttr => {
                let [name, class] = read_attrs(e, [b"name", b"class"])?;
                self.attr = (owned(name), owned(class));
                self.capture();
            }
            Kind::Plan => {
                let v = read_attrs(e, [b"selected", b"score"])?;
                if v[0].as_deref() == Some("yes") {
                    self.person.selected = self.person.n_plans;
                }
                self.person.n_plans += 1;
                let score = match non_empty(&v[1]) {
                    Some(s) => pyfloat(s)?,
                    None => f64::NAN,
                };
                self.plan = PlanState { score, ..Default::default() };
            }
            Kind::Act => self.activity(e)?,
            Kind::Leg => {
                if self.plan.acts.is_empty() {
                    bail!("plan starts with a leg");
                }
                let [mode, trav_time] = read_attrs(e, [b"mode", b"trav_time"])?;
                self.leg = LegInfo {
                    mode: owned(mode),
                    trav_time: owned(trav_time),
                    ..Default::default()
                };
            }
            Kind::LegAttrs => {
                read_attrs(e, [])?;
                self.leg.attrs_seen = true;
            }
            Kind::LegAttr => {
                let [name] = read_attrs(e, [b"name"])?;
                // only the first routingMode with text counts
                if self.leg.routing.is_none() && name.as_deref() == Some("routingMode") {
                    self.capture();
                }
            }
            Kind::Route => {
                let [typ, start, end] = read_attrs(e, [b"type", b"start_link", b"end_link"])?;
                self.leg.route_seen = true;
                if self.keep_routes {
                    // (without keep_routes the route only marks the leg's first <route>)
                    self.leg.route = Some(RouteInfo {
                        typ: owned(typ),
                        start: owned(start),
                        end: owned(end),
                        text: String::new(),
                    });
                    self.capture();
                }
            }
            Kind::Root => unreachable!(),
            Kind::Other => {
                read_attrs(e, [])?;
            }
        }
        self.stack.push(kind);
        Ok(())
    }

    fn capture(&mut self) {
        self.text = Some(Text { depth: self.stack.len() + 1, open: true, buf: String::new() });
    }

    fn take_text(&mut self) -> String {
        match self.text.take() {
            Some(t) if t.depth == self.stack.len() + 1 => t.buf,
            other => {
                self.text = other;
                String::new()
            }
        }
    }

    fn data(&mut self, s: &str) -> R<()> {
        match &mut self.text {
            Some(t) if t.open && t.depth == self.stack.len() => t.buf.push_str(s),
            _ => {
                // text in the document element but not in a person: ignored
            }
        }
        Ok(())
    }

    fn end(&mut self) -> R<()> {
        let kind = self.stack.pop().ok_or("unbalanced end tag")?;
        match kind {
            Kind::Root => bail!("unbalanced chunk"),
            Kind::Person => {
                if self.person.n_plans == 0 {
                    bail!("person without plans");
                }
                let p = std::mem::take(&mut self.person);
                self.out.n_plans.push(p.n_plans);
                self.out.selected.push(p.selected);
                if !p.v4.is_empty() || !p.entries.is_empty() {
                    self.out.attrs.push(((self.out.ids.len() - 1) as i64, p.v4, p.entries));
                }
            }
            Kind::PersonAttr => {
                let text = self.take_text();
                let (name, class) = std::mem::take(&mut self.attr);
                let value = attribute_value(class.as_deref(), text)?;
                self.person.entries.push((name, value));
            }
            Kind::LegAttr => {
                // a capture started here only for a routingMode (else this is "")
                let text = self.take_text();
                if !text.is_empty() {
                    self.leg.routing = Some(text.trim().to_string());
                }
            }
            Kind::Route => {
                if self.keep_routes {
                    let text = self.take_text();
                    self.leg.route.as_mut().unwrap().text = text;
                }
            }
            Kind::Leg => {
                let leg = std::mem::take(&mut self.leg);
                self.plan.trip.push(Some(leg));
            }
            Kind::Plan => self.end_plan()?,
            _ => {}
        }
        Ok(())
    }

    fn activity(&mut self, e: &BytesStart) -> R<()> {
        let [typ, x, y, node, link, end_time, dur, max_dur] = read_attrs(
            e, [b"type", b"x", b"y", b"node", b"link", b"end_time", b"dur", b"max_dur"])?;
        if typ.as_deref().unwrap_or("").ends_with(" interaction") {
            self.plan.trip.push(None);
            return Ok(());
        }
        if !self.plan.trip.is_empty() {
            let trip = std::mem::take(&mut self.plan.trip);
            self.plan.legs.push(trip_leg(trip)?);
        } else if !self.plan.acts.is_empty() {
            bail!("two activities without a leg between");
        }
        let typ = typ.ok_or("activity without type")?;
        let typ = code(&mut self.out.type_code, &mut self.out.act_types, &typ);
        let (x, y) = match (&x, &y) {
            (Some(x), Some(y)) => (pyfloat(x)?, pyfloat(y)?),
            _ => (f64::NAN, f64::NAN),
        };
        let mut link_index = -1;
        let node_index = if let Some(node) = &node {
            *self.net.node.get(node.as_ref()).ok_or("unknown node")?
        } else if let Some(link) = &link {
            link_index = self.net.link_index(link)?;
            let n = self.net.to_node[link_index as usize];
            i32::try_from(n).map_err(|_| "node index")?
        } else {
            if x.is_nan() || y.is_nan() {
                bail!("activity without location");
            }
            -1
        };
        let end = match non_empty(&end_time) {
            Some(t) => parse_time(t)?,
            None => f64::NAN,
        };
        let dur = match non_empty(&dur).or(non_empty(&max_dur)) {
            Some(t) => parse_time(t)?,
            None => f64::NAN,
        };
        self.plan.acts.push(Act { typ, node: node_index, end, dur, x, y, link: link_index });
        Ok(())
    }

    fn end_plan(&mut self) -> R<()> {
        let plan = std::mem::take(&mut self.plan);
        if plan.acts.is_empty() {
            bail!("plan without activities");
        }
        // a trip after the last activity is dropped, as in the Python reader
        let mut acts = plan.acts.into_iter();
        self.out.act(acts.next().unwrap());
        for (leg, act) in plan.legs.into_iter().zip(acts) {
            match leg {
                TripLeg::Mode(mode) => self.out.leg(&mode, f64::NAN, None),
                TripLeg::File(leg) => self.out.file_leg(&leg, self.net, self.keep_routes)?,
            }
            self.out.act(act);
        }
        self.out.act_offsets.push(self.out.a_type.len() as i64);
        self.out.scores.push(plan.score);
        Ok(())
    }
}

/// Parses a chunk of whole children of the document element.
fn parse_chunk(chunk: &[u8], net: &Net, keep_routes: bool) -> R<Columns> {
    let mut reader = Reader::from_str(utf8(chunk)?);
    reader.config_mut().check_end_names = true;
    let mut p = Parser {
        net,
        keep_routes,
        out: Columns::new(),
        stack: vec![Kind::Root],
        person: PersonState::default(),
        plan: PlanState::default(),
        leg: LegInfo::default(),
        attr: (None, None),
        text: None,
    };
    loop {
        match reader.read_event().map_err(|e| format!("XML: {e}"))? {
            Event::Start(e) => p.start(&e)?,
            Event::Empty(e) => {
                p.start(&e)?;
                p.end()?;
            }
            Event::End(_) => p.end()?,
            Event::Text(t) => {
                if p.text.is_some() {
                    let s = utf8(&t)?;
                    p.data(&unescape(newlines(s))?)?;
                } else {
                    // not kept, but entity references must still be valid
                    let s = utf8(&t)?;
                    if s.contains('&') {
                        unescape(Cow::Borrowed(s))?;
                    }
                }
            }
            Event::CData(c) => p.data(&newlines(utf8(&c)?))?,
            Event::Comment(_) | Event::PI(_) => {}
            Event::Decl(_) | Event::DocType(_) => bail!("declaration in the document element"),
            Event::Eof => break,
        }
    }
    if p.stack.len() != 1 {
        bail!("unbalanced chunk");
    }
    Ok(p.out)
}

/// Bytes per chunk (then cut before the next `<person`), by default.
const CHUNK: usize = 1 << 21;

struct Splitter<S: Read> {
    src: S,
    buf: Vec<u8>,
}

impl<S: Read> Splitter<S> {
    /// Reads more; false at the end of the input.
    fn fill(&mut self) -> R<bool> {
        let old = self.buf.len();
        self.buf.resize(old + (1 << 20), 0);
        let n = loop {
            match self.src.read(&mut self.buf[old..]) {
                Ok(n) => break n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => bail!("read: {e}"),
            }
        };
        self.buf.truncate(old + n);
        Ok(n > 0)
    }
}

fn check_space(b: &[u8]) -> R<()> {
    if b.iter().all(|&c| is_space(c)) {
        Ok(())
    } else {
        bail!("text outside the document element")
    }
}

/// Reads the prolog and the document element's start tag at the start of
/// `buf` (UTF-8, no other encoding, no internal DTD subset, no namespaces).
/// Returns the element's name, where its content starts, and whether it is
/// empty (`<population/>`).
fn prolog(buf: &[u8]) -> R<(Vec<u8>, usize, bool)> {
    let bom = if buf.starts_with(b"\xEF\xBB\xBF") { 3 } else { 0 };
    let mut reader = Reader::from_reader(&buf[bom..]);
    loop {
        let event = reader.read_event().map_err(|e| format!("XML: {e}"))?;
        let (e, empty) = match event {
            Event::Start(e) => (e, false),
            Event::Empty(e) => (e, true),
            Event::Decl(d) => {
                if let Some(enc) = d.encoding() {
                    let enc = enc.map_err(|e| format!("XML: {e}"))?;
                    if !enc.eq_ignore_ascii_case(b"utf-8") && !enc.eq_ignore_ascii_case(b"utf8") {
                        bail!("encoding");
                    }
                }
                continue;
            }
            Event::DocType(d) => {
                if d.contains(&b'[') {
                    bail!("DTD internal subset");
                }
                continue;
            }
            Event::Text(t) => {
                check_space(&t)?;
                continue;
            }
            Event::Comment(_) | Event::PI(_) => continue,
            _ => bail!("no document element at the start"),
        };
        let end = bom + reader.buffer_position() as usize;
        utf8(&buf[..end])?;
        let name = e.name().as_ref().to_vec();
        if memchr::memchr(b':', &name).is_some() {
            bail!("namespaced element");
        }
        read_attrs(&e, [])?;
        return Ok((name, end, empty));
    }
}

/// Reads `src` and calls `send` with its chunks of persons, in order:
/// the content of the document element, cut before a `<person` about every
/// `chunk` bytes. Checks the document outside that content (see `prolog`;
/// after the element only spaces).
fn split<S: Read>(src: S, chunk: usize, mut send: impl FnMut(Vec<u8>)) -> R<()> {
    let mut s = Splitter { src, buf: Vec::with_capacity(4 << 20) };
    while s.buf.len() < 1 << 20 && s.fill()? {}
    let (root, start, empty) = prolog(&s.buf)?;
    let mut start = start;
    let finder = memmem::Finder::new(b"<person");
    let mut from = start + chunk;
    let mut eof = false;
    while !empty {
        // the next cut: a "<person" (then a space, '>' or '/') at or after `from`
        while !eof && s.buf.len() < from + 8 {
            eof = !s.fill()?;
        }
        if s.buf.len() < from + 8 {
            break;
        }
        match finder.find(&s.buf[from..]) {
            Some(k) if s.buf.len() >= from + k + 8 => {
                let at = from + k;
                if matches!(s.buf[at + 7], b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/') {
                    send(s.buf[start..at].to_vec());
                    s.buf.drain(..at);
                    start = 0;
                    from = chunk.max(1);
                } else {
                    from = at + 1;
                }
            }
            _ => from = s.buf.len().saturating_sub(7).max(from),
        }
        if from + 8 > s.buf.len() && !eof {
            eof = !s.fill()?;
        }
    }
    while s.fill()? {}
    let end = s.buf.iter().rposition(|&c| !is_space(c)).map_or(0, |k| k + 1);
    if empty {
        return check_space(&s.buf[start..]);
    }
    // the content ends with the document element's end tag (then only spaces)
    let close = memmem::rfind(&s.buf[start..end], b"</").ok_or("no end tag")? + start;
    let tag = &s.buf[close + 2..end];
    if tag.last() != Some(&b'>') || tag[..tag.len() - 1].trim_ascii_end() != root.as_slice() {
        bail!("the document element does not end the file");
    }
    send(s.buf[start..close].to_vec());
    Ok(())
}

/// Joins chunks' columns in order, renumbering activity types and modes in
/// order of first use over the whole file (as one `PlanBatchBuilder` does).
fn merge(parts: Vec<Columns>) -> Columns {
    let mut out = Columns::new();
    for c in parts {
        let types: Vec<i32> = c
            .act_types
            .iter()
            .map(|t| code(&mut out.type_code, &mut out.act_types, t))
            .collect();
        let modes: Vec<i32> =
            c.modes.iter().map(|m| code(&mut out.mode_code, &mut out.modes, m)).collect();
        let (a0, r0, p0) =
            (out.a_type.len() as i64, out.route_links.len() as i64, out.ids.len() as i64);
        out.act_offsets.extend(c.act_offsets[1..].iter().map(|o| o + a0));
        out.a_type.extend(c.a_type.iter().map(|&t| types[t as usize]));
        out.a_node.extend(c.a_node);
        out.a_end.extend(c.a_end);
        out.a_dur.extend(c.a_dur);
        out.a_x.extend(c.a_x);
        out.a_y.extend(c.a_y);
        out.a_link.extend(c.a_link);
        out.l_mode.extend(c.l_mode.iter().map(|&m| modes[m as usize]));
        out.l_tt.extend(c.l_tt);
        out.l_routed.extend(c.l_routed);
        out.l_rtt.extend(c.l_rtt);
        out.l_rdist.extend(c.l_rdist);
        out.route_offsets.extend(c.route_offsets[1..].iter().map(|o| o + r0));
        out.route_links.extend(c.route_links);
        out.ids.extend(c.ids);
        out.n_plans.extend(c.n_plans);
        out.selected.extend(c.selected);
        out.scores.extend(c.scores);
        out.attrs.extend(c.attrs.into_iter().map(|(i, v4, e)| (i + p0, v4, e)));
    }
    out
}

/// Splits `src` and parses the chunks on `threads` threads.
fn read<S: Read>(src: S, net: &Net, keep_routes: bool, threads: usize, chunk: usize) -> R<Columns> {
    let (job_tx, job_rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(2 * threads);
    let job_rx = Mutex::new(job_rx);
    let (res_tx, res_rx) = mpsc::channel::<(usize, R<Columns>)>();
    let split_result = std::thread::scope(|scope| {
        for _ in 0..threads {
            let (job_rx, res_tx) = (&job_rx, res_tx.clone());
            scope.spawn(move || loop {
                let job = job_rx.lock().unwrap().recv();
                let Ok((k, chunk)) = job else { break };
                if res_tx.send((k, parse_chunk(&chunk, net, keep_routes))).is_err() {
                    break;
                }
            });
        }
        drop(res_tx);
        let mut k = 0;
        let r = split(src, chunk, |bytes| {
            // a send fails only if all workers stopped (then the results say why)
            let _ = job_tx.send((k, bytes));
            k += 1;
        });
        drop(job_tx);
        r
    });
    split_result?;
    let mut parts: Vec<(usize, R<Columns>)> = res_rx.into_iter().collect();
    parts.sort_by_key(|(k, _)| *k);
    let parts = parts.into_iter().map(|(_, c)| c).collect::<R<Vec<Columns>>>()?;
    Ok(if parts.len() == 1 { parts.into_iter().next().unwrap() } else { merge(parts) })
}

fn read_file(path: &str, net: &Net, keep_routes: bool, threads: usize, chunk: usize) -> R<Columns> {
    let file = File::open(path).map_err(|e| format!("{e}"))?;
    if path.ends_with(".gz") {
        read(MultiGzDecoder::new(file), net, keep_routes, threads, chunk)
    } else {
        read(file, net, keep_routes, threads, chunk)
    }
}

/// The columns of a MATSim population file (see the module docs); raises
/// `FallBack` where the Python reader must read it. `threads`: 0 for up to
/// 8 (the cores); `chunk_bytes`: the size of the parse jobs.
#[pyfunction]
#[pyo3(signature = (path, link_ids, node_ids, to_node, free_speed_travel_times, length, keep_routes, threads=0, chunk_bytes=CHUNK))]
#[allow(clippy::too_many_arguments)]
fn read_population_columns<'py>(
    py: Python<'py>,
    path: String,
    link_ids: Vec<Option<String>>,
    node_ids: Vec<Option<String>>,
    to_node: Vec<i64>,
    free_speed_travel_times: Vec<f64>,
    length: Vec<f64>,
    keep_routes: bool,
    threads: usize,
    chunk_bytes: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let threads = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(8)
    };
    let net = Net {
        link: id_map(link_ids),
        node: id_map(node_ids),
        to_node,
        fstt: free_speed_travel_times,
        length,
    };
    let c = py
        .allow_threads(|| read_file(&path, &net, keep_routes, threads, chunk_bytes))
        .map_err(FallBack::new_err)?;
    let d = PyDict::new(py);
    d.set_item("ids", c.ids)?;
    d.set_item("n_plans", c.n_plans.into_pyarray(py))?;
    d.set_item("selected", c.selected.into_pyarray(py))?;
    d.set_item("scores", c.scores.into_pyarray(py))?;
    d.set_item("attributes", c.attrs)?;
    d.set_item("act_types", c.act_types)?;
    d.set_item("modes", c.modes)?;
    d.set_item("act_offsets", c.act_offsets.into_pyarray(py))?;
    d.set_item("act_type", c.a_type.into_pyarray(py))?;
    d.set_item("act_node", c.a_node.into_pyarray(py))?;
    d.set_item("act_end_time", c.a_end.into_pyarray(py))?;
    d.set_item("act_duration", c.a_dur.into_pyarray(py))?;
    d.set_item("act_x", c.a_x.into_pyarray(py))?;
    d.set_item("act_y", c.a_y.into_pyarray(py))?;
    d.set_item("act_link", c.a_link.into_pyarray(py))?;
    d.set_item("leg_mode", c.l_mode.into_pyarray(py))?;
    d.set_item("leg_travel_time", c.l_tt.into_pyarray(py))?;
    d.set_item("leg_routed", c.l_routed.into_pyarray(py))?;
    d.set_item("leg_route_time", c.l_rtt.into_pyarray(py))?;
    d.set_item("leg_route_distance", c.l_rdist.into_pyarray(py))?;
    d.set_item("route_offsets", c.route_offsets.into_pyarray(py))?;
    d.set_item("route_links", c.route_links.into_pyarray(py))?;
    Ok(d)
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FallBack", m.py().get_type::<FallBack>())?;
    m.add_function(wrap_pyfunction!(read_population_columns, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times() {
        assert_eq!(parse_time("01:02:03.5").unwrap(), 3723.5);
        assert_eq!(parse_time(" 25:00 ").unwrap(), 90000.0);
        assert!(parse_time("undefined").is_err());
        assert!(parse_time("-1:00").is_err()); // Python accepts it: falls back
    }

    #[test]
    fn truncated_chunks_are_errors() {
        let net = Net { link: HashMap::new(), node: HashMap::new(), to_node: vec![], fstt: vec![], length: vec![] };
        for chunk in ["<person id=\"a\"><plan>", "\n<!-- c ", "<person id=\"a", "<x><![CDATA[ y",
                      "<?pi x", "<person id='a'/></x>", "<x>"] {
            assert!(parse_chunk(chunk.as_bytes(), &net, false).is_err(), "{chunk}");
        }
        assert!(parse_chunk(b"\n<!-- c --><x/>\n", &net, false).is_ok());
    }

    #[test]
    fn attribute_normalisation() {
        assert_eq!(attr_value(b"a\r\nb\tc&amp;&#10;").unwrap(), "a b c&\n");
    }
}
