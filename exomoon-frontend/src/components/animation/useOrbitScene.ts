'use client';
import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import type { TrajectoryFrame, SimulationMeta } from '@/lib/types';
import { useSimulationStore } from '@/hooks/useSimulationStore';

// ── Tuning knobs ────────────────────────────────────────────────────────────
const TRAIL_MAX     = 600;    // max trail points per body in the circular buffer
const TARGET_FPS    = 60;

// Body visual radii expressed as fractions of the planet's semi-major axis.
const STAR_FRAC     = 0.05;
const PLANET_FRAC   = 0.02;
const MOON_FRAC     = 0.01;
const STAR_MIN_R    = 0.003;
const PLANET_MIN_R  = 0.002;
const MOON_MIN_R    = 0.001;

// HZ ring opacity
const HZ_OUTER_OPACITY = 0.13;
// ────────────────────────────────────────────────────────────────────────────

const STAR_BASE_R   = 0.06;
const PLANET_BASE_R = 0.025;
const MOON_BASE_R   = 0.012;

export interface SceneControls {
  frameIndex: number;
  totalFrames: number;
  isPlaying: boolean;
  speedMultiplier: number;
  setFrameIndex: (i: number) => void;
  setIsPlaying: (v: boolean) => void;
  setSpeedMultiplier: (v: number) => void;
}

export interface BodyRadiiAU {
  star: number;
  planet: number;
  moon: number;
}

export function useOrbitScene(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  frames: TrajectoryFrame[] | null,
  meta: SimulationMeta | null,
  bodyRadii?: BodyRadiiAU
): SceneControls {
  const [frameIndex, setFrameIndexState] = useState(0);
  const [isPlaying, setIsPlayingState] = useState(false);
  const [speedMultiplier, setSpeedMultiplierState] = useState(1);

  const frameIndexRef = useRef(0);
  const isPlayingRef  = useRef(false);
  const speedRef      = useRef(1);
  const framesRef     = useRef<TrajectoryFrame[] | null>(null);

  const rendererRef      = useRef<THREE.WebGLRenderer | null>(null);
  const labelRendererRef = useRef<CSS2DRenderer | null>(null);
  const sceneRef         = useRef<THREE.Scene | null>(null);
  const cameraRef        = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef      = useRef<OrbitControls | null>(null);
  const rafRef           = useRef<number | null>(null);

  const starRef   = useRef<THREE.Mesh | null>(null);
  const planetRef = useRef<THREE.Mesh | null>(null);
  const moonRef   = useRef<THREE.Mesh | null>(null);

  // Three world-space trails: [star (yellow), planet (blue), moon (red)]
  // Ring buffers are written sequentially; display buffers are reordered copies.
  // Separating them eliminates the jump artifact where Three.js draws a line
  // from the newest point back to position[0] when the ring wraps.
  const trailRingBufs  = useRef<Float32Array[]>([]);   // raw ring (write here)
  const trailGeomRefs  = useRef<THREE.BufferGeometry[]>([]);
  const trailLinesRef  = useRef<THREE.Line[]>([]);
  const trailHeadsRef  = useRef<number[]>([0, 0, 0]);  // next write index per trail
  const trailFillsRef  = useRef<number[]>([0, 0, 0]);  // valid point count per trail

  const hzOuterRef = useRef<THREE.Mesh | null>(null);
  const hzInnerRef = useRef<THREE.Mesh | null>(null);

  // ML stability annulus (purple shell between predicted am_min and am_max)
  const mlShellRef = useRef<THREE.Mesh | null>(null);

  const setFrameIndex      = useCallback((i: number) => { frameIndexRef.current = i; setFrameIndexState(i); }, []);
  const setIsPlaying       = useCallback((v: boolean) => { isPlayingRef.current = v; setIsPlayingState(v); }, []);
  const setSpeedMultiplier = useCallback((v: number) => { speedRef.current = v; setSpeedMultiplierState(v); }, []);

  // ── ML store state ───────────────────────────────────────────────────────────
  const { mlPrediction, mlMassIdx } = useSimulationStore();

  // ── Scene initialisation ─────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setClearColor(0x050a14);
    rendererRef.current = renderer;

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    labelRenderer.domElement.style.pointerEvents = 'none';
    canvas.parentElement?.appendChild(labelRenderer.domElement);
    labelRendererRef.current = labelRenderer;

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 5, 5);
    scene.add(dirLight);
    sceneRef.current = scene;

    // Background stars
    const starGeo = new THREE.BufferGeometry();
    const starPositions = new Float32Array(6000);
    for (let i = 0; i < 6000; i++) starPositions[i] = (Math.random() - 0.5) * 200;
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.08 })));

    const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.0001, 500);
    camera.position.set(0, 3, 6);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 0.001;
    controls.maxDistance = 200;
    controlsRef.current = controls;

    const mkSphere = (r: number, color: number) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(r, 24, 24),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.3 })
      );
      scene.add(m);
      return m;
    };
    starRef.current   = mkSphere(STAR_BASE_R,   0xFFDD00);
    planetRef.current = mkSphere(PLANET_BASE_R, 0x4488FF);
    moonRef.current   = mkSphere(MOON_BASE_R,   0xFF5555);

    const mkLabel = (text: string, color: string) => {
      const div = document.createElement('div');
      div.textContent = text;
      div.style.cssText = `font-size:10px;color:${color};font-family:monospace;pointer-events:none;`;
      return new CSS2DObject(div);
    };
    starRef.current.add(mkLabel('★ Star', '#FFD700'));
    planetRef.current.add(mkLabel('● Planet', '#88AAFF'));
    moonRef.current.add(mkLabel('◦ Moon', '#FF8888'));

    // Three world-space trails: star, planet, moon.
    // Each trail has a ring buffer (write) and a display buffer (ordered copy).
    // The display buffer is what Three.js renders — always in chronological order
    // (oldest → newest), so there is no jump segment when the ring wraps around.
    const trailColors = [0xFFDD00, 0x4488FF, 0xFF5555];
    const trailOpacities = [0.40, 0.40, 0.50];
    trailColors.forEach((color, idx) => {
      const ringBuf = new Float32Array(TRAIL_MAX * 3).fill(0);
      const dispBuf = new Float32Array(TRAIL_MAX * 3).fill(0);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(dispBuf, 3));
      geo.setDrawRange(0, 0);
      const line = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color, opacity: trailOpacities[idx], transparent: true })
      );
      scene.add(line);
      trailRingBufs.current[idx] = ringBuf;
      trailGeomRefs.current[idx] = geo;
      trailLinesRef.current[idx] = line;
      trailHeadsRef.current[idx] = 0;
      trailFillsRef.current[idx] = 0;
    });

    // Resize observer
    const ro = new ResizeObserver(() => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      renderer.setSize(w, h, false);
      labelRenderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(canvas.parentElement!);

    // RAF loop
    let lastTime = 0;
    const tick = (timestamp: number) => {
      rafRef.current = requestAnimationFrame(tick);
      const delta = timestamp - lastTime;
      if (delta < 1000 / TARGET_FPS) return;
      lastTime = timestamp;

      const f = framesRef.current;
      if (f && f.length > 0) {
        if (isPlayingRef.current) {
          const next = frameIndexRef.current + speedRef.current;
          const clamped = next >= f.length ? 0 : next;
          frameIndexRef.current = clamped;
          setFrameIndexState(Math.floor(clamped));
        }
        updateScene(Math.floor(frameIndexRef.current));
      }

      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      labelRenderer.domElement.remove();
    };
  }, [canvasRef]);

  // ── Update scene when frames / meta change ───────────────────────────────
  useEffect(() => {
    framesRef.current = frames;

    if (!frames || frames.length === 0) return;

    // Reset playback and all trail state
    frameIndexRef.current = 0;
    setFrameIndexState(0);
    trailHeadsRef.current = [0, 0, 0];
    trailFillsRef.current = [0, 0, 0];
    trailRingBufs.current.forEach(b => b.fill(0));
    trailGeomRefs.current.forEach(g => {
      g.setDrawRange(0, 0);
      const attr = g.attributes.position as THREE.BufferAttribute;
      (attr.array as Float32Array).fill(0);
      attr.needsUpdate = true;
    });

    // ── Resize body spheres ──────────────────────────────────────────────────
    let apEst = 0;
    for (const f of frames) {
      const d = Math.sqrt((f.planet_x - f.star_x) ** 2 + (f.planet_y - f.star_y) ** 2);
      if (d > apEst) apEst = d;
    }
    if (apEst > 0) {
      let starR: number, planetR: number, moonR: number;
      if (bodyRadii && bodyRadii.star > 0) {
        const maxActual = Math.max(bodyRadii.star, bodyRadii.planet, bodyRadii.moon);
        const targetMax = Math.max(apEst * STAR_FRAC, STAR_MIN_R);
        const sf = targetMax / maxActual;
        starR   = Math.max(bodyRadii.star   * sf, STAR_MIN_R);
        planetR = Math.max(bodyRadii.planet * sf, PLANET_MIN_R);
        moonR   = Math.max(bodyRadii.moon   * sf, MOON_MIN_R);
      } else {
        starR   = Math.max(apEst * STAR_FRAC,   STAR_MIN_R);
        planetR = Math.max(apEst * PLANET_FRAC, PLANET_MIN_R);
        moonR   = Math.max(apEst * MOON_FRAC,   MOON_MIN_R);
      }
      if (starRef.current)   starRef.current.scale.setScalar(starR   / STAR_BASE_R);
      if (planetRef.current) planetRef.current.scale.setScalar(planetR / PLANET_BASE_R);
      if (moonRef.current)   moonRef.current.scale.setScalar(moonR   / MOON_BASE_R);
    }

    // ── HZ shells ──────────────────────────────────────────────────────────
    const scene = sceneRef.current;
    if (!scene || !meta) return;

    if (hzOuterRef.current) {
      scene.remove(hzOuterRef.current);
      hzOuterRef.current.geometry.dispose();
      (hzOuterRef.current.material as THREE.Material).dispose();
      hzOuterRef.current = null;
    }
    if (hzInnerRef.current) {
      scene.remove(hzInnerRef.current);
      hzInnerRef.current.geometry.dispose();
      (hzInnerRef.current.material as THREE.Material).dispose();
      hzInnerRef.current = null;
    }

    const { a_inner_au, a_outer_au } = meta;

    // ShaderMaterial on the outer sphere: for each surface fragment, cast a ray
    // from the camera through the fragment and check if it first intersects the
    // inner sphere.  Fragments whose ray hits the inner sphere are discarded —
    // they're "above" the interior, not the shell — so only the true shell
    // (between a_inner_au and a_outer_au) is rendered, with no interior tint.
    const shellMat = new THREE.ShaderMaterial({
      uniforms: {
        uInnerR2: { value: a_inner_au * a_inner_au },
        uColor:   { value: new THREE.Color(0x00cc44) },
        uOpacity: { value: HZ_OUTER_OPACITY },
      },
      vertexShader: /* glsl */`
        varying vec3 vWorldPos;
        void main() {
          vec4 wp   = modelMatrix * vec4(position, 1.0);
          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */`
        uniform float uInnerR2;
        uniform vec3  uColor;
        uniform float uOpacity;
        varying vec3  vWorldPos;
        void main() {
          // Ray from camera toward this outer-sphere surface point.
          // Inner sphere is centred at world origin.
          vec3  oc   = cameraPosition;
          vec3  dir  = normalize(vWorldPos - cameraPosition);
          float b    = dot(oc, dir);
          float c    = dot(oc, oc) - uInnerR2;
          float disc = b * b - c;
          // disc > 0  →  ray intersects inner sphere.
          // t1 = -b - sqrt(disc) > 0  →  entry point is in front of camera,
          // i.e. the inner sphere lies between the camera and this fragment.
          if (disc > 0.0 && (-b - sqrt(disc)) > 0.0) discard;
          gl_FragColor = vec4(uColor, uOpacity);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const hzShell = new THREE.Mesh(
      new THREE.SphereGeometry(a_outer_au, 48, 48),
      shellMat,
    );
    scene.add(hzShell);
    hzOuterRef.current = hzShell;

    // Auto-fit camera
    const allX = frames.map(f => Math.max(Math.abs(f.star_x), Math.abs(f.planet_x), Math.abs(f.moon_x)));
    const maxExtent = Math.max(...allX, a_outer_au) * 1.5;
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, maxExtent * 0.6, maxExtent * 1.2);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }

  }, [frames, meta]);

  // ── Per-frame scene update ───────────────────────────────────────────────
  const updateScene = useCallback((idx: number) => {
    const f = framesRef.current;
    if (!f || idx >= f.length) return;
    const frame = f[idx];

    // All three bodies: star (0), planet (1), moon (2)
    const bodies = [
      { mesh: starRef.current,   x: frame.star_x,   y: frame.star_y,   z: frame.star_z },
      { mesh: planetRef.current, x: frame.planet_x, y: frame.planet_y, z: frame.planet_z },
      { mesh: moonRef.current,   x: frame.moon_x,   y: frame.moon_y,   z: frame.moon_z },
    ];

    bodies.forEach(({ mesh, x, y, z }, bi) => {
      if (!mesh) return;
      mesh.position.set(x, y, z);

      const ring = trailRingBufs.current[bi];
      const geo  = trailGeomRefs.current[bi];
      if (!ring || !geo) return;

      const head = trailHeadsRef.current[bi];
      const fill = trailFillsRef.current[bi];

      // Write new position to ring buffer
      ring[head * 3]     = x;
      ring[head * 3 + 1] = y;
      ring[head * 3 + 2] = z;

      const newHead = (head + 1) % TRAIL_MAX;
      const newFill = Math.min(fill + 1, TRAIL_MAX);
      trailHeadsRef.current[bi] = newHead;
      trailFillsRef.current[bi] = newFill;

      // Copy ring → display buffer in chronological order (oldest → newest).
      // When the ring hasn't wrapped yet (newFill < TRAIL_MAX), oldest is index 0.
      // After wrapping, oldest is newHead (the slot about to be overwritten next).
      // This reordering removes the line-jump artifact that occurs when Three.js
      // draws positions[0..N] in buffer order across a wrap boundary.
      const attr = geo.attributes.position as THREE.BufferAttribute;
      const disp = attr.array as Float32Array;
      const oldestIdx = newFill < TRAIL_MAX ? 0 : newHead;
      for (let i = 0; i < newFill; i++) {
        const srcIdx = (oldestIdx + i) % TRAIL_MAX;
        disp[i * 3]     = ring[srcIdx * 3];
        disp[i * 3 + 1] = ring[srcIdx * 3 + 1];
        disp[i * 3 + 2] = ring[srcIdx * 3 + 2];
      }
      attr.needsUpdate = true;
      geo.setDrawRange(0, newFill);
    });
  }, []);

  // ── ML stability annulus ─────────────────────────────────────────────────────
  // When the user moves the moon-mass slider in MlMapOverlay the shell updates
  // in real time to show the valid orbital-radius band for that mass (Hill radii
  // converted to AU using the current simulation's rhill_AU from meta).
  // Uses the identical ShaderMaterial as the HZ shell but violet (0x8b5cf6).
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Clean up any previous ML shell
    if (mlShellRef.current) {
      scene.remove(mlShellRef.current);
      mlShellRef.current.geometry.dispose();
      (mlShellRef.current.material as THREE.Material).dispose();
      mlShellRef.current = null;
    }

    // Need both a prediction and a rhill_AU from the current sim to draw the shell
    if (!mlPrediction || !meta?.rhill_AU) return;

    const amRange = mlPrediction.validAmPerMm[mlMassIdx];
    if (!amRange) return;   // no valid orbit at this mass

    const rhill      = meta.rhill_AU;
    const amInnerAU  = amRange[0] * rhill;
    const amOuterAU  = amRange[1] * rhill;

    if (amOuterAU <= amInnerAU || amOuterAU <= 0) return;

    // Reuse the same GLSL shader as the HZ shell — only uColor changes
    const shellMat = new THREE.ShaderMaterial({
      uniforms: {
        uInnerR2: { value: amInnerAU * amInnerAU },
        uColor:   { value: new THREE.Color(0x8b5cf6) },   // violet-500
        uOpacity: { value: 0.18 },
      },
      vertexShader: /* glsl */`
        varying vec3 vWorldPos;
        void main() {
          vec4 wp   = modelMatrix * vec4(position, 1.0);
          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */`
        uniform float uInnerR2;
        uniform vec3  uColor;
        uniform float uOpacity;
        varying vec3  vWorldPos;
        void main() {
          vec3  oc   = cameraPosition;
          vec3  dir  = normalize(vWorldPos - cameraPosition);
          float b    = dot(oc, dir);
          float c    = dot(oc, oc) - uInnerR2;
          float disc = b * b - c;
          if (disc > 0.0 && (-b - sqrt(disc)) > 0.0) discard;
          gl_FragColor = vec4(uColor, uOpacity);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    const mlShell = new THREE.Mesh(
      new THREE.SphereGeometry(amOuterAU, 48, 48),
      shellMat,
    );
    scene.add(mlShell);
    mlShellRef.current = mlShell;

  }, [mlPrediction, mlMassIdx, meta]);

  return {
    frameIndex,
    totalFrames: frames?.length ?? 0,
    isPlaying,
    speedMultiplier,
    setFrameIndex,
    setIsPlaying,
    setSpeedMultiplier,
  };
}
