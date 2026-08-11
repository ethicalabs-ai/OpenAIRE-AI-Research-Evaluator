<script setup>
import { ref, onMounted, watch, onUnmounted } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls'
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer'
import { gsap } from 'gsap'

const povActive = ref(false)
let pointerLockControls
let moveForward = false
let moveBackward = false
let moveLeft = false
let moveRight = false
const velocity = new THREE.Vector3()
const direction = new THREE.Vector3()

const resetCamera = () => {
  if (!camera || !controls) return
  gsap.to(camera.position, { x: 50, y: 30, z: 50, duration: 1.0, ease: 'power2.inOut' })
  gsap.to(controls.target, { x: 0, y: 0, z: 0, duration: 1.0, ease: 'power2.inOut' })
}

const togglePOV = () => {
  if (!pointerLockControls) return
  if (povActive.value) {
    pointerLockControls.unlock()
  } else {
    pointerLockControls.lock()
  }
}

const props = defineProps({
  traceStep: Object,
  traceIndex: Number,
  numLayers: { type: Number, default: 8 },
  onLog: { type: Function, default: () => {} }
})

const emit = defineEmits(['selectVoxel', 'selectGantry', 'deselect'])

const canvasRef = ref(null)
const labelContainerRef = ref(null)

let scene, camera, renderer, labelRenderer, controls, clock
let inputBlock, outputHead, gantries = [], stateStacks = [], fluxLines = []
let pulse, pulseGroup
let entropySphere, entropyHalo, entropyGlow
let filamentGroup
let lastIndex = -1
let raycaster = new THREE.Raycaster()
let mouse = new THREE.Vector2()

// --- Interaction Logic ---
let pointerDownPos = new THREE.Vector2()
const onPointerDown = (event) => {
  pointerDownPos.set(event.clientX, event.clientY)
}

let lastSelectedHash = null

const performRaycast = (mouseVec) => {
  if (!renderer || !camera) return
  raycaster.setFromCamera(mouseVec, camera)
  raycaster.params.Line.threshold = 1.0

  const allVoxels = stateStacks.flatMap(s => s.voxels)
  const allGantries = gantries.map(g => g.frame)
  const entropyObj = entropySphere
  const intersects = raycaster.intersectObjects([...allVoxels, ...allGantries, entropyObj, ...entropySphere.children, ...filamentGroup.children])

  if (intersects.length > 0) {
    let selected = intersects[0].object

    // If we hit halo or glow (children of entropySphere), use the parent
    if (selected.parent === entropySphere) {
      selected = entropySphere
    }

    // Hierarchical Hit-Testing: Find the parent if we hit a wireframe/child
    const isGantry = selected.userData.type === 'gantry'
    const isEntropy = selected.userData.type === 'entropy'
    const isFilament = selected.userData.type === 'filament'

    if (!isGantry && !isEntropy && !isFilament && selected.userData.layerIndex === undefined && selected.parent && selected.parent.userData.layerIndex !== undefined) {
      selected = selected.parent
    }

    let currentHash = selected.uuid
    if (isEntropy) currentHash = 'entropy'
    else if (isGantry) currentHash = `gantry_${selected.userData.layerIndex}_${selected.userData.voxelIndex}`
    else if (isFilament) currentHash = `filament_${selected.userData.layerIndex}_${selected.userData.intensity}`
    else currentHash = `voxel_${selected.userData.layerIndex}_${selected.userData.voxelIndex}`

    if (povActive.value && lastSelectedHash === currentHash) return
    lastSelectedHash = currentHash

    // Emit selection event
    if (isEntropy) {
      const entropy = props.traceStep?.entropy ?? 0
      const description = 'Latent State Entropy: Measures distributional uncertainty ' +
        'from next-token logits via Shannon entropy H(p) = -Σ p(x)log(p(x)). ' +
        'Higher entropy = more uncertainty, Lower entropy = more confident predictions.'
      emit('selectVoxel', { type: 'entropy', entropy, name: 'Entropy Core', description })
      emit('select-voxel', { type: 'entropy', entropy, name: 'Entropy Core', description })
    } else if (isGantry) {
      const payload = {
        layerIndex: selected.userData.layerIndex,
        voxelIndex: selected.userData.voxelIndex,
        gridCoord: selected.userData.gridCoord,
        type: selected.userData.type || 'gantry'
      }
      emit('selectGantry', payload)
      emit('select-gantry', payload)
    } else if (isFilament) {
      const payload = {
        layerIndex: selected.userData.layerIndex,
        intensity: selected.userData.intensity,
        type: 'filament'
      }
      emit('selectVoxel', payload)
      emit('select-voxel', payload)
    } else {
      const payload = {
        layerIndex: selected.userData.layerIndex,
        voxelIndex: selected.userData.voxelIndex,
        gridCoord: selected.userData.gridCoord,
        type: selected.userData.type || 'voxel'
      }
      emit('selectVoxel', payload)
      emit('select-voxel', payload)
    }
  } else {
    if (povActive.value && lastSelectedHash === null) return
    lastSelectedHash = null
    // Clicked on empty space - emit deselect
    emit('deselect')
  }
}

const onPointerUp = (event) => {
  if (!renderer || !camera) return
  const dist = Math.hypot(event.clientX - pointerDownPos.x, event.clientY - pointerDownPos.y)
  if (dist > 5) return

  const rect = renderer.domElement.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  // If POV is active, manual clicks are disabled to prevent conflict with auto-raycast
  if (povActive.value) return

  performRaycast(mouse)
}

const initScene = () => {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
  // Adjusted for circular layout - view from side to see the arc
  camera.position.set(50, 30, 50)

  renderer = new THREE.WebGLRenderer({ canvas: canvasRef.value, antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

  labelRenderer = new CSS2DRenderer()
  labelRenderer.setSize(window.innerWidth, window.innerHeight)
  labelRenderer.domElement.style.position = 'absolute'
  labelRenderer.domElement.style.top = '0px'
  labelRenderer.domElement.style.pointerEvents = 'none'
  document.getElementById('app').appendChild(labelRenderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
  scene.add(ambientLight)

  const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x000000, 0.6)
  scene.add(hemisphereLight)

  const pointLight = new THREE.PointLight(0xffae00, 1, 150)
  pointLight.position.set(20, 30, 20)
  scene.add(pointLight)

  const gridHelper = new THREE.GridHelper(200, 50, 0x333333, 0x111111)
  scene.add(gridHelper)

  // 1. Input Block
  const geometry = new THREE.BoxGeometry(3, 3, 3)
  const material = new THREE.MeshPhongMaterial({ color: 0xffae00, emissive: 0xffae00, emissiveIntensity: 0.5, wireframe: true })
  inputBlock = new THREE.Mesh(geometry, material)
  scene.add(inputBlock)
  createLabel(inputBlock, "INPUT EMBEDDING", 10)

  // 1b. Output Head
  outputHead = new THREE.Mesh(geometry, material)
  scene.add(outputHead)
  createLabel(outputHead, "OUTPUT HEAD", 10)

  // 1c. Entropy Sphere (Black Hole with Reddish Halo)
  const coreRadius = 2
  const sphereGeo = new THREE.SphereGeometry(coreRadius, 32, 32)
  const sphereMat = new THREE.MeshPhongMaterial({ 
    color: 0x000000, 
    transparent: true,
    depthWrite: false,
    opacity: 0.85,
    emissive: 0x000000,
    emissiveIntensity: 0
  })
  entropySphere = new THREE.Mesh(sphereGeo, sphereMat)
  entropySphere.position.set(0, 0, 0)
  entropySphere.userData = { type: 'entropy', name: 'Entropy Core' }
  scene.add(entropySphere)
  entropySphere.scale.set(1, 1, 1)

  // Reddish halo at fixed distance (doesn't scale)
  const haloRadius = 2.8
  const haloGeo = new THREE.SphereGeometry(haloRadius, 32, 32)
  const haloMat = new THREE.MeshPhongMaterial({ 
    color: 0xff3300, 
    transparent: true,
    depthWrite: false,
    opacity: 0.35,
    emissive: 0xff3300,
    emissiveIntensity: 0.6,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending
  })
  entropyHalo = new THREE.Mesh(haloGeo, haloMat)
  entropySphere.add(entropyHalo)

  // Inner reddish glow for depth (fixed size)
  const glowRadius = 2.5
  const glowGeo = new THREE.SphereGeometry(glowRadius, 32, 32)
  const glowMat = new THREE.MeshBasicMaterial({ 
    color: 0xff6600, 
    transparent: true, 
    opacity: 0.25,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  })
  entropyGlow = new THREE.Mesh(glowGeo, glowMat)
  entropySphere.add(entropyGlow)

  // Label
  createLabel(entropySphere, "ENTROPY", 4)

  setupLayers(props.numLayers)

  // 3. Neural Spark (Pulse Group + Trails)
  pulseGroup = new THREE.Group()
  scene.add(pulseGroup)
  
  // Filament Group for Attention
  filamentGroup = new THREE.Group()
  scene.add(filamentGroup)

  const pulseGeom = new THREE.SphereGeometry(1.2, 32, 32)
  const pulseMat = new THREE.MeshStandardMaterial({ color: 0xffae00, emissive: 0xffae00, emissiveIntensity: 2 })
  pulse = new THREE.Mesh(pulseGeom, pulseMat)
  pulseGroup.add(pulse)

  const pulseLight = new THREE.PointLight(0xffae00, 2, 10)
  pulse.add(pulseLight)


  pulseGroup.visible = false
  clock = new THREE.Clock()
  animate()
}

const createLabel = (object, text, yOffset = 10) => {
  const div = document.createElement('div')
  div.className = 'scene-label'
  div.textContent = text
  div.style.color = '#ffffff'
  div.style.fontFamily = "'JetBrains Mono', monospace"
  div.style.fontSize = '12px'
  div.style.padding = '2px 5px'
  div.style.background = 'rgba(0,0,0,0.8)'
  div.style.border = '1px solid #ffffff'
  const label = new CSS2DObject(div)
  label.position.set(0, yOffset, 0)
  object.add(label)
}

const setupLayers = (count) => {
  gantries.forEach(g => scene.remove(g.group))
  stateStacks.forEach(s => scene.remove(s.group))
  fluxLines.forEach(l => scene.remove(l))
  gantries = []; stateStacks = []; fluxLines = []

  // Circular layout configuration - full circle
  const radius = 35 // Circle radius
  const centerX = 0 // X-center of circle
  const centerY = 0 // Y-center of circle (for perfect centering)
  const startAngle = 0 // Start at right (0 degrees)
  const endAngle = Math.PI * 2 // Full circle (360 degrees)
  const totalAngle = endAngle - startAngle

  // Position input at start of circle (angle 0, right side)
  const inputAngle = 0
  const inputX = centerX + radius * Math.cos(inputAngle)
  const inputZ = centerY + radius * Math.sin(inputAngle)
  inputBlock.position.set(inputX, centerY, inputZ)
  inputBlock.visible = true

  // Calculate positions for all layers along the full circle
  // Total items: 1 input + count layers + 1 output = count + 2 items
  // Gantry positions: at midpoints between consecutive items (where pipes were)
  // Stack positions: at layer positions
  const totalSlots = count + 2
  const itemSpacing = totalAngle / totalSlots // spacing between consecutive items

  for (let i = 0; i < count; i++) {
    // Gantry is at midpoint between layer i and layer i+1
    const gantrySlot = i + 1
    const gantryAngle = gantrySlot * itemSpacing
    const gantryX = centerX + radius * Math.cos(gantryAngle)
    const gantryZ = centerY + radius * Math.sin(gantryAngle)

    const gantryGroup = new THREE.Group()
    gantryGroup.position.set(gantryX, centerY, gantryZ)
    // Rotate gantry to be perpendicular to circle (face tangent)
    gantryGroup.rotation.y = -gantryAngle + Math.PI / 2

    const ringGeom = new THREE.TorusGeometry(6.5, 0.4, 6, 6)
    ringGeom.rotateY(Math.PI / 2)
    const ringMat = new THREE.MeshPhongMaterial({ color: 0x666666 })
    const ring = new THREE.Mesh(ringGeom, ringMat)
    ring.userData = { type: 'gantry', layerIndex: i }
    gantryGroup.add(ring)

    const blades = []
    const bladeGeom = new THREE.BoxGeometry(0.2, 5.5, 3.5)
    const bladeMat = new THREE.MeshPhongMaterial({ color: 0x444444, transparent: true, opacity: 0.8, depthWrite: false })
    for (let j = 0; j < 6; j++) {
      const blade = new THREE.Mesh(bladeGeom, bladeMat)
      const angle = (j / 6) * Math.PI * 2
      blade.position.set(0, Math.sin(angle) * 3, Math.cos(angle) * 3)
      blade.rotation.x = angle
      gantryGroup.add(blade)
      blades.push(blade)
    }
    scene.add(gantryGroup)
    gantries.push({ group: gantryGroup, blades, frame: ring })

    // Label positioned radially outward from the circle
    const labelRadius = radius + 8
    const labelAngle = gantryAngle
    const labelX = centerX + labelRadius * Math.cos(labelAngle)
    const labelZ = centerY + labelRadius * Math.sin(labelAngle)
    const labelGroup = new THREE.Group()
    labelGroup.position.set(labelX, centerY, labelZ)
    scene.add(labelGroup)
    createLabel(labelGroup, `UNIT L${i}`, 10)

    // Stack is at the layer position (between two gantries)
    const stackSlot = i + 0.5
    const stackAngle = stackSlot * itemSpacing
    const stackX = centerX + radius * Math.cos(stackAngle)
    const stackZ = centerY + radius * Math.sin(stackAngle)

    const stackGroup = new THREE.Group()
    stackGroup.position.set(stackX, centerY, stackZ)
    // Rotate stack to be perpendicular to circle (face tangent)
    stackGroup.rotation.y = -stackAngle + Math.PI / 2

    const voxels = []
    for(let x=0; x<5; x++) {
      for(let y=0; y<5; y++) {
        for(let z=0; z<5; z++) {
          const vMesh = new THREE.Mesh(
            new THREE.BoxGeometry(0.8, 0.8, 0.8),
            new THREE.MeshPhongMaterial({ color: 0x888888, emissive: 0x333333, transparent: true, opacity: 0.1, depthWrite: false })
          )
          const edges = new THREE.LineSegments(new THREE.EdgesGeometry(vMesh.geometry), new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.4, depthWrite: false }))
          vMesh.add(edges)
          vMesh.userData = { wireframe: edges, layerIndex: i, voxelIndex: voxels.length, gridCoord: { x,y,z } }
          edges.userData = { ...vMesh.userData }

          vMesh.position.set(x*1.3 - 2.6, y*1.3 - 2.6, z*1.3 - 2.6)
          stackGroup.add(vMesh)
          voxels.push(vMesh)
        }
      }
    }

    scene.add(stackGroup)
    stateStacks.push({ group: stackGroup, voxels })
  }

  // Position output head closer to the last unit (L7)
  // Place it at slot count + 0.5 (halfway between last layer and original output position)
  const outputSlot = count + 0.5
  const outputAngle = outputSlot * itemSpacing
  const outputX = centerX + radius * Math.cos(outputAngle)
  const outputZ = centerY + radius * Math.sin(outputAngle)
  outputHead.position.set(outputX, centerY, outputZ)
  outputHead.visible = true
}

const animate = () => {
  requestAnimationFrame(animate)
  const dt = clock.getDelta()
  // Cap delta to prevent velocity explosions if the tab goes inactive
  const delta = Math.min(dt, 0.1)

  if (povActive.value) {
    direction.z = Number(moveForward) - Number(moveBackward)
    direction.x = Number(moveRight) - Number(moveLeft)
    
    if (direction.lengthSq() > 0) {
      direction.normalize()
    }

    if (moveForward || moveBackward) velocity.z -= direction.z * 400.0 * delta
    if (moveLeft || moveRight) velocity.x -= direction.x * 400.0 * delta

    // Damping
    velocity.x -= velocity.x * 10.0 * delta
    velocity.z -= velocity.z * 10.0 * delta

    // Use raw camera translations instead of pointerLockControls to allow free 3D flight (pitch up to fly up)
    // local Z axis points backwards (+Z is backwards). So we translate by positive velocity.z (which is negative) to go forwards.
    camera.translateX(-velocity.x * delta)
    camera.translateZ(velocity.z * delta)
    
    // Auto raycast from center of screen in POV
    const centerMouse = new THREE.Vector2(0, 0)
    performRaycast(centerMouse)
  } else {
    controls.update()
  }

  renderer.render(scene, camera)
  labelRenderer.render(scene, camera)
}

const updateVisuals = () => {
  const step = props.traceStep
  if (!step) return

  // Prevent Animation Overlap
  gsap.killTweensOf(pulseGroup.position)
  pulseGroup.children.forEach(c => gsap.killTweensOf(c))
  
  // Clear old filaments
  while(filamentGroup.children.length > 0){ 
      const child = filamentGroup.children[0]
      filamentGroup.remove(child)
      if (child.geometry) child.geometry.dispose()
      if (child.material) child.material.dispose()
  }

  const isBackwards = props.traceIndex < lastIndex
  lastIndex = props.traceIndex

  const triggerLayer = (layerIdx) => {
    const layerData = step.layers[layerIdx]
    if (!layerData) return

    const gantry = gantries[layerIdx]
    const stack = stateStacks[layerIdx]
    if (!gantry || !stack) return

    // Aperture Shift
    const offset = (layerData.lambda > 0.1) ? (layerData.lambda > 0.5 ? 5 : 3.5) : 1.5
    gantry.blades.forEach((b, j) => {
      const a = (j/6)*Math.PI*2
      gsap.to(b.position, { y: Math.sin(a)*offset, z: Math.cos(a)*offset, duration: 0.2 })
    })
    const gateColor = (layerData.lambda > 0.5) ? new THREE.Color(1, 0.1, 0) : new THREE.Color(0.4, 0.4, 0.4)
    gsap.to(gantry.frame.material.color, { r: gateColor.r, g: gateColor.g, b: gateColor.b, duration: 0.3 })

    // Voxel Flare
    stack.voxels.forEach((v, vIdx) => {
      const val = Math.abs(layerData.c_state[(vIdx*4)%layerData.c_state.length] || 0)
      const intensity = Math.min(val*15, 1)
      const color = new THREE.Color().setHSL(intensity < 0.6 ? 0.08 : 0.02, 1, 0.3 + intensity*0.2)
      v.material.color.set(color); v.material.emissive.set(color)
      gsap.to(v.material, { opacity: 0.15 + intensity * 0.45, emissiveIntensity: 0.2 + intensity * 1.5, duration: 0.4 })
      gsap.to(v.scale, { x: 1+intensity*0.8, y: 1+intensity*0.8, z: 1+intensity*0.8, duration: 0.4 })
    })



    // Synaptic Filaments (Attention) (persist permanently for step)
    if (layerData.attention && layerData.attention.length > 0) {
        layerData.attention.forEach(attn => {
            if (attn.val < 0.05) return;
            const targetPos = stack.group.position.clone()
            targetPos.y += (Math.random() - 0.5) * 4
            targetPos.x += (Math.random() - 0.5) * 4
            const sourcePos = inputBlock.position.clone() 
            const midPos = new THREE.Vector3().addVectors(sourcePos, targetPos).multiplyScalar(0.5)
            midPos.y += 15 
            const curve = new THREE.QuadraticBezierCurve3(sourcePos, midPos, targetPos)
            const points = curve.getPoints(20)
            const geom = new THREE.BufferGeometry().setFromPoints(points)
            const mat = new THREE.LineBasicMaterial({ 
                color: 0xffaa00, transparent: true, opacity: 0.0,
                blending: THREE.AdditiveBlending, depthWrite: false
            })
            const line = new THREE.Line(geom, mat)
            line.userData = { type: 'filament', layerIndex: layerIdx, intensity: attn.val }
            filamentGroup.add(line)
            const intensity = Math.min(attn.val * 3, 1)
            gsap.to(mat, { opacity: intensity, duration: 0.2, ease: "power2.inOut" })
        })
    }
  }

  const triggerOutputHead = () => {
    if (!step.output_head) return

    if (step.output_head.attention && step.output_head.attention.length > 0) {
      step.output_head.attention.forEach(attn => {
          if (attn.val < 0.05) return;
          const targetPos = outputHead.position.clone()
          targetPos.y += (Math.random() - 0.5) * 4
          targetPos.x += (Math.random() - 0.5) * 4
          
          let sourcePos = inputBlock.position.clone()
          // Valid layer index provided? Route from that layer's stack.
          if (attn.idx !== undefined && stateStacks[attn.idx]) {
             sourcePos = stateStacks[attn.idx].group.position.clone()
          }
          
          const midPos = new THREE.Vector3().addVectors(sourcePos, targetPos).multiplyScalar(0.5)
          midPos.y += 15 
          const curve = new THREE.QuadraticBezierCurve3(sourcePos, midPos, targetPos)
          const points = curve.getPoints(20)
          const geom = new THREE.BufferGeometry().setFromPoints(points)
          const mat = new THREE.LineBasicMaterial({ 
              color: 0x00e5ff, // Cyan for output-bound context
              transparent: true, opacity: 0.0,
              blending: THREE.AdditiveBlending, depthWrite: false
          })
          const line = new THREE.Line(geom, mat)
          line.userData = { type: 'filament', layerIndex: 'OUT', intensity: attn.val }
          filamentGroup.add(line)
          const intensity = Math.min(attn.val * 3, 1)
          gsap.to(mat, { opacity: intensity, duration: 0.2, ease: "power2.inOut" })
      })
    }
  }

  if (isBackwards) {
    pulseGroup.visible = false
    gantries.forEach((g, i) => triggerLayer(i))
    triggerOutputHead()
    return
  }

  // Forward 'Nice' Spline Flow along circular arc
  gsap.to(inputBlock.material, { emissiveIntensity: 2, duration: 0.2, yoyo: true, repeat: 1 })
  pulseGroup.visible = true

  // Construct Spline Path along circular arc
  const points = [inputBlock.position.clone()]
  gantries.forEach((g, i) => {
    points.push(g.group.position.clone())
    points.push(stateStacks[i].group.position.clone())
  })
  points.push(outputHead.position.clone())
  const spline = new THREE.CatmullRomCurve3(points)
  spline.closed = false // Open curve from top to bottom

  const flowObj = { progress: 0 }
  const triggeredLayers = new Set()

  gsap.to(flowObj, {
    progress: 1,
    duration: 1.2, // Smoother speed
    ease: "power2.inOut",
    onUpdate: () => {
      const p = spline.getPoint(flowObj.progress)
      pulseGroup.position.copy(p)

      // Trigger unit effects as pulse passes
      const layerIdx = Math.min(Math.floor(flowObj.progress * props.numLayers), props.numLayers - 1)
      if (layerIdx >= 0 && !triggeredLayers.has(layerIdx)) {
        triggeredLayers.add(layerIdx)
        triggerLayer(layerIdx)
      }
    },
    onComplete: () => {
      triggerOutputHead()
      if (step.is_generated) {
        gsap.to(outputHead.material, { emissiveIntensity: 3, duration: 0.2, yoyo: true, repeat: 1 })
        props.onLog(`TOKEN EMITTED: "${step.token}"`)
      }
      gsap.to(pulseGroup, { visible: false, duration: 0.2 })
    }
  })
}

watch(() => props.traceIndex, (idx) => {
  updateVisuals()
  // Use entropy from backend (Shannon entropy from logits)
  const entropy = props.traceStep?.entropy ?? 0
  const scale = 0.5 + (entropy / 1.2) * 1.5 // Map entropy 0-1.2 to scale 0.5-2.0
  gsap.to(entropySphere.scale, { x: scale, y: scale, z: scale, duration: 0.3 })
  // Halo and glow stay at fixed size (don't scale)
})
watch(() => props.numLayers, (n) => setupLayers(n))

onMounted(() => {
  initScene()
  window.addEventListener('resize', onResize)
  renderer.domElement.addEventListener('pointerdown', onPointerDown)
  renderer.domElement.addEventListener('pointerup', onPointerUp)

  // Configure POV Controls
  pointerLockControls = new PointerLockControls(camera, document.body)
  scene.add(pointerLockControls.getObject())

  pointerLockControls.addEventListener('lock', () => { povActive.value = true })
  pointerLockControls.addEventListener('unlock', () => { 
    povActive.value = false
    resetCamera() 
  })

  const onKeyDown = (event) => {
    switch (event.code) {
      case 'ArrowUp':
      case 'KeyW': moveForward = true; break;
      case 'ArrowLeft':
      case 'KeyA': moveLeft = true; break;
      case 'ArrowDown':
      case 'KeyS': moveBackward = true; break;
      case 'ArrowRight':
      case 'KeyD': moveRight = true; break;
    }
  }

  const onKeyUp = (event) => {
    switch (event.code) {
      case 'ArrowUp':
      case 'KeyW': moveForward = false; break;
      case 'ArrowLeft':
      case 'KeyA': moveLeft = false; break;
      case 'ArrowDown':
      case 'KeyS': moveBackward = false; break;
      case 'ArrowRight':
      case 'KeyD': moveRight = false; break;
    }
  }

  document.addEventListener('keydown', onKeyDown)
  document.addEventListener('keyup', onKeyUp)

  onUnmounted(() => {
    document.removeEventListener('keydown', onKeyDown)
    document.removeEventListener('keyup', onKeyUp)
    if (renderer) renderer.dispose()
    if (labelRenderer) document.getElementById('app').removeChild(labelRenderer.domElement)
  })
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  if (labelRenderer) document.getElementById('app').removeChild(labelRenderer.domElement)
})

const onResize = () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
  labelRenderer.setSize(window.innerWidth, window.innerHeight)
}
</script>

<template>
  <div>
    <canvas ref="canvasRef"></canvas>
    <div v-if="povActive" class="crosshair">+</div>
    <button :class="['pov-toggle-btn', { active: povActive }]" @click="togglePOV">
      {{ povActive ? '[ POV ON ]' : '[ POV OFF ]' }}
    </button>
  </div>
</template>

<style scoped>
canvas { position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; }

.crosshair {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: rgba(255, 255, 255, 0.7);
  font-family: monospace;
  font-size: 24px;
  pointer-events: none;
  z-index: 10;
}

.pov-toggle-btn {
  position: absolute;
  top: 20px;
  right: 20px;
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: rgba(255, 255, 255, 0.6);
  padding: 8px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  font-weight: bold;
  letter-spacing: 1px;
  cursor: pointer;
  z-index: 1000;
  transition: all 0.2s ease;
  border-radius: 4px;
}

.pov-toggle-btn:hover {
  border-color: rgba(255, 255, 255, 0.5);
  color: #fff;
}

.pov-toggle-btn.active {
  border-color: #00e5ff;
  color: #00e5ff;
  box-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
}
</style>