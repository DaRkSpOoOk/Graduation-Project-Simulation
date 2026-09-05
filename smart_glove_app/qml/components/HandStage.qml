import QtQuick
import QtQuick3D
import QtQuick3D.AssetUtils

// TASK-007G hero stage.
//
// Three concerns, deliberately kept apart:
//
//   1. PRESENTATION  - where each hand sits on screen, its base orientation and
//                      the palm/back view. These are plain QML bindings on nodes
//                      this file owns, so recorded data can never write to them
//                      and playing a sign can never move the layout.
//   2. ARTICULATION  - per-bone rotations solved from the frozen channels and
//                      applied inside each loaded hand.
//   3. CAMERA        - derived from the layout and the viewport aspect, so the
//                      framing is identical at every window size. There is no
//                      accumulating orbit state in normal operation.
//
// Each hand is a separate GLB in its own RuntimeLoader: Qt Quick 3D mis-binds a
// glTF carrying two skins, deforming the second mesh with the first skeleton.
Item {
    id: root

    property url leftAssetUrl
    property url rightAssetUrl
    property bool assetAvailable: false
    property var rigProfile: ({})
    property var handPose: ({})

    property string viewMode: "PALM"          // PALM | BACK
    property string materialMode: "SKIN"      // SKIN | GLOVE | WIREFRAME
    property bool debugManoPoints: false
    property var leftGeometry
    property var rightGeometry

    property string leftState: "IDLE"
    property string rightState: "IDLE"
    property bool leftDimmed: false
    property bool rightDimmed: false

    property bool inspectMode: false          // opt-in orbit, for debugging only
    property real inspectYaw: 0.0
    property real inspectPitch: 0.0
    property real zoom: 1.0

    readonly property bool rigReady: leftHand.ready && rightHand.ready
    readonly property bool rigFailed: leftHand.failed || rightHand.failed
    readonly property string rigError: leftHand.failed ? leftHand.error : rightHand.error

    readonly property var _presentation: rigProfile.presentation || ({})
    readonly property real fieldOfView: _presentation.camera_field_of_view_deg || 35.0
    readonly property real contentHalfWidth: _presentation.content_half_width || 0.90
    readonly property real contentHalfHeight: _presentation.content_half_height || 0.68

    // ---- camera framing -----------------------------------------------------
    // Distance is solved from the content box so the hands stay the same size on
    // screen whatever the window does: the camera is locked to the presentation
    // layout rather than being a free viewer.
    readonly property real _aspect: view3d.height > 0 ? view3d.width / view3d.height : 1.6
    readonly property real _halfFovTan: Math.tan(fieldOfView * Math.PI / 360.0)
    readonly property real fitDistance: {
        var byHeight = contentHalfHeight / _halfFovTan
        var byWidth = contentHalfWidth / (_halfFovTan * Math.max(0.2, _aspect))
        return Math.max(byHeight, byWidth) / Math.max(0.35, zoom)
    }

    function resetView() {
        root.inspectYaw = 0.0
        root.inspectPitch = 0.0
        root.zoom = 1.0
        root.inspectMode = false
    }

    function viewEuler() {
        var views = root._presentation.views || ({})
        var entry = views[root.viewMode]
        if (entry && entry.root_euler_deg)
            return Qt.vector3d(entry.root_euler_deg[0], entry.root_euler_deg[1], entry.root_euler_deg[2])
        return root.viewMode === "BACK" ? Qt.vector3d(0, 180, 0) : Qt.vector3d(0, 0, 0)
    }

    function sidePosition(side) {
        var key = side === "LEFT" ? "left_position" : "right_position"
        var value = root._presentation[key]
        if (value)
            return Qt.vector3d(value[0], value[1], value[2])
        return Qt.vector3d(side === "LEFT" ? -0.55 : 0.55, 0, 0)
    }

    function applyPose(pose) {
        leftHand.applyPose(pose ? pose["LEFT"] : null)
        rightHand.applyPose(pose ? pose["RIGHT"] : null)
    }

    function reportStatus() {
        if (!appState)
            return
        if (root.rigReady)
            appState.setRigAssetStatus("Hands ready — left and right skeletons resolved")
        else if (root.rigFailed)
            appState.setRigAssetStatus("Hand asset error — " + root.rigError)
    }

    onHandPoseChanged: applyPose(handPose)
    onRigReadyChanged: reportStatus()
    onRigFailedChanged: reportStatus()

    Rectangle {
        anchors.fill: parent
        radius: 14
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#151a22" }
            GradientStop { position: 0.55; color: "#0d1117" }
            GradientStop { position: 1.0; color: "#080a10" }
        }
    }

    // One rigged hand: its own GLB, its own skeleton, its own presentation node.
    component RiggedHand: Node {
        id: hand
        property string side: "LEFT"
        property url assetUrl
        property var sceneIndex: ({})
        property int requiredBoneCount: 0
        property var material
        property bool ready: false
        property bool failed: false
        property string error: ""

        property var _bones: ({})
        property var _rest: ({})
        property var _models: ([])
        property int _attempts: 0

        function nodeAtPath(container, path) {
            var current = container
            for (var i = 0; i < path.length; ++i) {
                var children = current ? current.children : null
                if (!children)
                    return null
                var step = Number(path[i])
                if (!(step >= 0 && step < children.length))
                    return null
                current = children[step]
            }
            return current
        }

        function collect(node, sink, modelsOnly, depth) {
            if (!node || depth > 9)
                return
            if (!modelsOnly || node instanceof Model)
                sink.push(node)
            var children = node.children || []
            for (var i = 0; i < children.length; ++i)
                collect(children[i], sink, modelsOnly, depth + 1)
        }

        // Try one candidate container: accept it only if every required bone
        // resolves to a distinct node. RuntimeLoader drops the glTF node names,
        // so the child-index paths come from the shipped GLB itself and the
        // subtree they are relative to is validated here rather than assumed.
        function bind(container) {
            var entry = hand.sceneIndex
            if (!entry || !entry.bones)
                return null
            var bones = ({}), rest = ({}), seen = ([])
            for (var name in entry.bones) {
                var bone = nodeAtPath(container, entry.bones[name])
                if (!bone || seen.indexOf(bone) !== -1)
                    return null
                seen.push(bone)
                bones[name] = bone
                // Snapshot the rest rotation as plain numbers. `bone.rotation`
                // hands back a live reference to the property, so caching the
                // value type itself would make every frame compose onto the
                // previous frame's result and accumulate without bound - which
                // is exactly what made a played sign crumple the hand and
                // appear to orbit.
                var r = bone.rotation
                rest[name] = [r.scalar, r.x, r.y, r.z]
            }
            if (seen.length !== hand.requiredBoneCount)
                return null
            return { bones: bones, rest: rest }
        }

        function resolve() {
            if (loader.status !== RuntimeLoader.Success)
                return
            var candidates = ([])
            collect(loader, candidates, false, 0)
            var bound = null
            for (var i = 0; i < candidates.length && !bound; ++i)
                bound = bind(candidates[i])

            if (!bound) {
                hand._attempts += 1
                if (hand._attempts < 40) {
                    retry.restart()
                    return
                }
                hand.failed = true
                hand.ready = false
                hand.error = hand.side + ": could not locate the hand skeleton"
                return
            }

            var models = ([])
            collect(loader, models, true, 0)
            hand._models = models
            hand._bones = bound.bones
            hand._rest = bound.rest
            hand.failed = false
            hand.error = ""
            hand.ready = true
            applyMaterial()
            applyPose(root.handPose ? root.handPose[hand.side] : null)
        }

        function applyMaterial() {
            if (!hand.ready || !hand.material)
                return
            for (var i = 0; i < hand._models.length; ++i)
                hand._models[i].materials = [hand.material]
        }

        function applyPose(sidePose) {
            if (!hand.ready || !sidePose)
                return
            var deltas = sidePose.bones || ({})
            for (var name in deltas) {
                var bone = hand._bones[name]
                var rest = hand._rest[name]
                var q = deltas[name]
                if (!bone || !rest || !q || q.length !== 4)
                    continue
                // Always rebuilt from the immutable rest snapshot, so the pose
                // is absolute per frame and never compounds.
                bone.rotation = Qt.quaternion(rest[0], rest[1], rest[2], rest[3])
                                  .times(Qt.quaternion(q[0], q[1], q[2], q[3]))
            }
        }

        onMaterialChanged: applyMaterial()

        RuntimeLoader {
            id: loader
            source: hand.assetUrl
            visible: hand.ready
            onStatusChanged: {
                if (status === RuntimeLoader.Success) {
                    hand._attempts = 0
                    retry.restart()
                } else if (status === RuntimeLoader.Error) {
                    hand.failed = true
                    hand.ready = false
                    hand.error = hand.side + ": " + errorString
                }
            }
        }

        Timer { id: retry; interval: 40; repeat: false; onTriggered: hand.resolve() }
    }

    View3D {
        id: view3d
        anchors.fill: parent
        anchors.margins: 1
        renderMode: View3D.Offscreen

        readonly property var activeMaterial: root.materialMode === "GLOVE" ? gloveMaterial
                                            : root.materialMode === "WIREFRAME" ? wireframeMaterial
                                            : skinMaterial

        // A plain SceneEnvironment on purpose. Enabling ANY ExtendedSceneEnvironment
        // post-processing pass here (tonemap, SSAO or vignette, in any
        // combination) composited the whole View3D down to a near-black
        // silhouette on this D3D11/RHI path - reproducible even with an unlit
        // material, so it is not a lighting problem. Shaping comes from the
        // light rig below instead.
        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Color
            clearColor: "#0e1219"
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
        }

        // Camera rig. The inspect pivot sits at identity unless the operator
        // explicitly turns inspect mode on, so no orbit can accumulate.
        Node {
            id: cameraPivot
            position: {
                var target = root._presentation.camera_look_at
                return target ? Qt.vector3d(target[0], target[1], target[2]) : Qt.vector3d(0, 0.02, 0)
            }
            eulerRotation: root.inspectMode
                ? Qt.vector3d(root.inspectPitch, root.inspectYaw, 0)
                : Qt.vector3d(0, 0, 0)

            PerspectiveCamera {
                id: camera
                position: Qt.vector3d(0, 0, root.fitDistance)
                fieldOfView: root.fieldOfView
                fieldOfViewOrientation: PerspectiveCamera.Vertical
                clipNear: 0.05
                clipFar: 60.0
            }
        }

        // Three-point rig tuned for skin: warm key, cool fill, cool rim, plus a
        // gentle frontal bounce so palm detail stays readable in PALM view.
        DirectionalLight {
            eulerRotation: Qt.vector3d(-28, -26, 0)
            color: "#fff4e8"
            brightness: 1.35
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(14, 44, 0)
            color: "#d6e4ff"
            brightness: 0.60
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(16, 176, 0)
            color: "#a8c6ff"
            brightness: 0.70
        }
        DirectionalLight {
            eulerRotation: Qt.vector3d(-6, 4, 0)
            color: "#ffeede"
            brightness: 0.45
        }

        PrincipledMaterial {
            id: skinMaterial
            baseColor: Qt.rgba(0.855, 0.663, 0.553, 1.0)
            roughness: 0.55
            metalness: 0.0
            specularAmount: 0.4
            clearcoatAmount: 0.05
            clearcoatRoughnessAmount: 0.5
            cullMode: PrincipledMaterial.NoCulling
        }
        PrincipledMaterial {
            id: gloveMaterial
            baseColor: "#2b323c"
            roughness: 0.68
            metalness: 0.08
            specularAmount: 0.35
            cullMode: PrincipledMaterial.NoCulling
        }
        PrincipledMaterial {
            id: wireframeMaterial
            baseColor: "#7fd0ff"
            roughness: 0.9
            metalness: 0.0
            lighting: PrincipledMaterial.NoLighting
            cullMode: PrincipledMaterial.NoCulling
        }

        // ---- presentation nodes: owned here, never written by recorded data --
        Node {
            id: leftPresentation
            position: root.sidePosition("LEFT")
            eulerRotation: root.viewEuler()
            opacity: root.leftDimmed ? 0.45 : 1.0
            Behavior on eulerRotation { Vector3dAnimation { duration: 260; easing.type: Easing.InOutCubic } }

            RiggedHand {
                id: leftHand
                side: "LEFT"
                assetUrl: root.leftAssetUrl
                sceneIndex: (root.rigProfile.sceneIndex && root.rigProfile.sceneIndex.sides)
                    ? root.rigProfile.sceneIndex.sides["LEFT"] : ({})
                requiredBoneCount: (root.rigProfile.required_bones || []).length
                material: view3d.activeMaterial
            }
        }

        Node {
            id: rightPresentation
            position: root.sidePosition("RIGHT")
            eulerRotation: root.viewEuler()
            opacity: root.rightDimmed ? 0.45 : 1.0
            Behavior on eulerRotation { Vector3dAnimation { duration: 260; easing.type: Easing.InOutCubic } }

            RiggedHand {
                id: rightHand
                side: "RIGHT"
                assetUrl: root.rightAssetUrl
                sceneIndex: (root.rigProfile.sceneIndex && root.rigProfile.sceneIndex.sides)
                    ? root.rigProfile.sceneIndex.sides["RIGHT"] : ({})
                requiredBoneCount: (root.rigProfile.required_bones || []).length
                material: view3d.activeMaterial
            }
        }

        // Optional MANO point diagnostic. Never part of normal playback.
        Model {
            visible: root.debugManoPoints && root.leftGeometry !== undefined
            geometry: root.leftGeometry
            position: root.sidePosition("LEFT")
            scale: Qt.vector3d(0.344, 0.344, 0.344)
            materials: [ DefaultMaterial { diffuseColor: "#78d0ff"; pointSize: 2.0 } ]
        }
        Model {
            visible: root.debugManoPoints && root.rightGeometry !== undefined
            geometry: root.rightGeometry
            position: root.sidePosition("RIGHT")
            scale: Qt.vector3d(0.344, 0.344, 0.344)
            materials: [ DefaultMaterial { diffuseColor: "#78d0ff"; pointSize: 2.0 } ]
        }
    }

    // ---- overlays -----------------------------------------------------------

    Rectangle {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 18
        width: 132; height: 30; radius: 15
        color: "#161c26"
        border.color: "#2b3644"; border.width: 1
        opacity: 0.92
        Text {
            anchors.centerIn: parent
            text: root.viewMode === "PALM" ? "PALM VIEW" : "BACK VIEW"
            color: "#8fb4dd"; font.pixelSize: 11; font.bold: true; font.letterSpacing: 1.2
        }
    }

    component SideTag: Rectangle {
        id: tag
        property string label: ""
        property bool dim: false
        width: 96; height: 34; radius: 17
        color: "#12171f"
        border.color: tag.dim ? "#5c3b30" : "#252f3c"
        border.width: 1
        opacity: 0.88
        Row {
            anchors.centerIn: parent
            spacing: 7
            Rectangle {
                width: 6; height: 6; radius: 3
                anchors.verticalCenter: parent.verticalCenter
                color: tag.dim ? "#e08b6a" : "#5ec98a"
            }
            Text {
                text: tag.label
                color: "#c7d4e3"; font.pixelSize: 11; font.bold: true; font.letterSpacing: 1.0
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }

    SideTag {
        anchors.left: parent.left; anchors.bottom: parent.bottom; anchors.margins: 18
        label: "LEFT"; dim: root.leftDimmed
    }
    SideTag {
        anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.margins: 18
        label: "RIGHT"; dim: root.rightDimmed
    }

    // Loading / failure notice. Quiet by design: it must never dominate the UI.
    Rectangle {
        visible: !root.rigReady
        anchors.centerIn: parent
        width: Math.min(parent.width - 60, 460)
        height: 56
        radius: 10
        color: "#141a23"
        border.color: root.rigFailed ? "#7a4436" : "#2b3644"
        border.width: 1
        Text {
            anchors.fill: parent
            anchors.margins: 12
            text: root.rigFailed
                ? "Hand asset error\n" + root.rigError
                : (root.assetAvailable ? "Preparing hands…"
                                       : "Hand assets not found — pass --rig-asset")
            color: root.rigFailed ? "#f0a58c" : "#93a4b8"
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            wrapMode: Text.Wrap
        }
    }

    // Inspect-only orbit. Off by default; pressing a letter can never enter it.
    MouseArea {
        anchors.fill: parent
        enabled: root.inspectMode
        acceptedButtons: Qt.LeftButton
        property real lastX: 0
        property real lastY: 0
        onPressed: (mouse) => { lastX = mouse.x; lastY = mouse.y }
        onPositionChanged: (mouse) => {
            if (!pressed)
                return
            root.inspectYaw += (mouse.x - lastX) * 0.30
            root.inspectPitch = Math.max(-40, Math.min(40, root.inspectPitch + (mouse.y - lastY) * 0.22))
            lastX = mouse.x
            lastY = mouse.y
        }
        onWheel: (wheel) => {
            root.zoom = Math.max(0.6, Math.min(2.2, root.zoom + wheel.angleDelta.y / 1600.0))
            wheel.accepted = true
        }
    }

    Connections {
        target: appState
        function onHandPoseChanged() { root.applyPose(appState.handPose) }
        function onResetViewRequested() { root.resetView() }
    }
}
