import QtQuick 6.8
import QtQuick3D 6.8
import QtQuick3D.AssetUtils 6.8

Item {
    id: root

    property var leftGeometry
    property var rightGeometry
    property var leftMarkers
    property var rightMarkers
    property bool surfaceMode: false
    property bool debugManoPoints: false
    property bool leftDimmed: false
    property bool rightDimmed: false
    property string leftState: "IDLE"
    property string rightState: "IDLE"
    property url rigAssetUrl
    property var rigProfile: ({})
    property var rigPose: ({})
    property bool rigAssetAvailable: false
    property bool rigAssetReady: false
    property bool rigAssetFailed: false
    property string rigAssetError: ""
    property var _boneNodes: ({})
    property var _baseRotations: ({})
    property var _presentationNodes: ({})
    property int _cacheAttempts: 0
    property real cameraDistance: 8.0
    property real cameraYaw: 0.0
    property real cameraPitch: -4.0

    function resetView() {
        cameraDistance = 8.0
        cameraYaw = 0.0
        cameraPitch = -4.0
    }

    function findNode(node, wantedName) {
        if (!node)
            return null
        if (node.objectName === wantedName || node.name === wantedName)
            return node
        var children = node.children
        if (!children)
            return null
        for (var index = 0; index < children.length; ++index) {
            var found = findNode(children[index], wantedName)
            if (found)
                return found
        }
        return null
    }

    function nodeAtPath(node, path) {
        if (!node || !path || path.length === 0)
            return null
        var current = node
        for (var index = 0; index < path.length; ++index) {
            var children = current.children || ([])
            var childIndex = Number(path[index])
            if (childIndex < 0 || childIndex >= children.length)
                return null
            current = children[childIndex]
        }
        return current
    }

    function cacheRigNodes() {
        if (!root.rigAssetAvailable || rigLoader.status !== RuntimeLoader.Success)
            return

        var roots = root.rigProfile.presentation_roots || ({})
        var armatures = root.rigProfile.armatures || ({})
        var runtimePaths = root.rigProfile.runtime_node_paths || ({})
        var required = root.rigProfile.required_deform_bones || ([])
        var nodes = ({})
        var bases = ({})
        var presentations = ({})
        var missing = ([])

        for (var sideIndex = 0; sideIndex < 2; ++sideIndex) {
            var side = sideIndex === 0 ? "LEFT" : "RIGHT"
            var sidePaths = runtimePaths[side] || ({})
            var presentation = nodeAtPath(rigLoader, sidePaths.presentation_root)
            if (!presentation)
                presentation = findNode(rigLoader, roots[side])
            var armature = nodeAtPath(rigLoader, sidePaths.armature)
            if (!armature)
                armature = findNode(presentation || rigLoader, armatures[side])
            presentations[side] = presentation
            nodes[side] = ({})
            bases[side] = ({})
            if (!presentation)
                missing.push(roots[side])
            if (!armature)
                missing.push(armatures[side])
            for (var boneIndex = 0; boneIndex < required.length; ++boneIndex) {
                var boneName = required[boneIndex]
                var bonePaths = sidePaths.bones || ({})
                var boneNode = nodeAtPath(rigLoader, bonePaths[boneName])
                if (!boneNode && armature)
                    boneNode = findNode(armature, boneName)
                if (!boneNode) {
                    missing.push(side + "/" + boneName)
                } else {
                    nodes[side][boneName] = boneNode
                    bases[side][boneName] = boneNode.rotation
                }
            }
        }

        if (missing.length > 0) {
            root._cacheAttempts += 1
            if (root._cacheAttempts < 20) {
                rigCacheTimer.restart()
                return
            }
            root.rigAssetFailed = true
            root.rigAssetError = "Rig profile nodes missing: " + missing.slice(0, 4).join(", ")
            root.rigAssetReady = false
            if (appState)
                appState.setRigAssetStatus("Rigged GLB rejected — " + root.rigAssetError)
            return
        }

        root._boneNodes = nodes
        root._baseRotations = bases
        root._presentationNodes = presentations
        root.rigAssetFailed = false
        root.rigAssetReady = true
        root.rigAssetError = ""
        if (appState)
            appState.setRigAssetStatus("Rigged GLB loaded — persistent skeleton ready")
        root.applyRigPose(root.rigPose)
    }

    function applyRigPose(pose) {
        if (!root.rigAssetReady || !pose)
            return
        for (var sideIndex = 0; sideIndex < 2; ++sideIndex) {
            var side = sideIndex === 0 ? "LEFT" : "RIGHT"
            var sidePose = pose[side]
            if (!sidePose)
                continue
            var sideNodes = root._boneNodes[side] || ({})
            var sideBases = root._baseRotations[side] || ({})
            var boneDeltas = sidePose.bones || ({})
            for (var boneName in boneDeltas) {
                var node = sideNodes[boneName]
                var baseRotation = sideBases[boneName]
                var values = boneDeltas[boneName]
                if (!node || !baseRotation || !values || values.length !== 4)
                    continue
                var delta = Qt.quaternion(Number(values[0]), Number(values[1]), Number(values[2]), Number(values[3]))
                node.rotation = baseRotation.times(delta)
            }
            var presentation = root._presentationNodes[side]
            if (presentation)
                presentation.opacity = sidePose.dimmed ? 0.43 : 1.0
        }
    }

    Rectangle {
        anchors.fill: parent
        color: "#111923"
        border.color: "#283443"
        border.width: 1
        radius: 10
    }

    View3D {
        id: scene3d
        anchors.fill: parent
        anchors.margins: 1

        environment: SceneEnvironment {
            clearColor: "#111923"
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
            depthTestEnabled: true
            aoEnabled: true
            aoStrength: 0.35
            aoDistance: 3.0
        }

        PerspectiveCamera {
            id: camera
            position: Qt.vector3d(0, 0, root.cameraDistance)
            eulerRotation: Qt.vector3d(root.cameraPitch, root.cameraYaw, 0)
            clipNear: 0.1
            clipFar: 80.0
        }

        DirectionalLight {
            eulerRotation: Qt.vector3d(-28, -26, 0)
            brightness: 1.25
            color: "#d7e5ff"
        }

        DirectionalLight {
            eulerRotation: Qt.vector3d(32, 150, 0)
            brightness: 0.45
            color: "#b9c8df"
        }

        PointLight {
            position: Qt.vector3d(0, 2.5, 3.0)
            brightness: 18
            color: "#c7d7ef"
            quadraticFade: 0.12
        }

        Node {
            id: handStage

            RuntimeLoader {
                id: rigLoader
                source: root.rigAssetAvailable ? root.rigAssetUrl : ""
                visible: root.rigAssetReady
                onStatusChanged: {
                    if (status === RuntimeLoader.Success) {
                        if (appState)
                            appState.setRigAssetStatus("Rigged GLB parsed - indexing persistent skeleton")
                        root._cacheAttempts = 0
                        rigCacheTimer.restart()
                    } else if (status === RuntimeLoader.Error) {
                        root.rigAssetFailed = true
                        root.rigAssetReady = false
                        root.rigAssetError = errorString
                        if (appState)
                            appState.setRigAssetStatus("Rigged GLB load failed — " + errorString)
                    }
                }
            }

            Model {
                id: leftSurface
                visible: !root.rigAssetReady && root.surfaceMode
                geometry: root.leftGeometry
                position: Qt.vector3d(-1.85, 0, 0)
                opacity: root.leftDimmed ? 0.52 : 1.0
                materials: [
                    PrincipledMaterial {
                        baseColor: root.leftDimmed ? "#35404e" : "#33485f"
                        roughness: 0.79
                        metalness: 0.06
                        specularAmount: 0.38
                        cullMode: PrincipledMaterial.BackFaceCulling
                    }
                ]
            }

            Model {
                id: rightSurface
                visible: !root.rigAssetReady && root.surfaceMode
                geometry: root.rightGeometry
                position: Qt.vector3d(1.85, 0, 0)
                opacity: root.rightDimmed ? 0.52 : 1.0
                materials: [
                    PrincipledMaterial {
                        baseColor: root.rightDimmed ? "#35404e" : "#33485f"
                        roughness: 0.79
                        metalness: 0.06
                        specularAmount: 0.38
                        cullMode: PrincipledMaterial.BackFaceCulling
                    }
                ]
            }

            // The old point representation is an explicit diagnostic view.
            // It is never enabled by normal asset playback.
            Model {
                id: leftPointCloud
                visible: root.debugManoPoints && !root.rigAssetReady
                geometry: root.leftGeometry
                position: Qt.vector3d(-1.85, 0, 0)
                opacity: root.leftDimmed ? 0.46 : 0.86
                materials: [
                    DefaultMaterial {
                        diffuseColor: root.leftDimmed ? "#596371" : "#7895b4"
                        pointSize: 2.0
                    }
                ]
            }

            Model {
                id: rightPointCloud
                visible: root.debugManoPoints && !root.rigAssetReady
                geometry: root.rightGeometry
                position: Qt.vector3d(1.85, 0, 0)
                opacity: root.rightDimmed ? 0.46 : 0.86
                materials: [
                    DefaultMaterial {
                        diffuseColor: root.rightDimmed ? "#596371" : "#7895b4"
                        pointSize: 2.0
                    }
                ]
            }

            Repeater3D {
                id: leftMarkerRepeater
                model: root.leftMarkers
                delegate: Model {
                    source: "#Sphere"
                    position: root.rigAssetReady
                        ? Qt.vector3d(model.position.x * 0.60 - 1.60, model.position.y * 0.60, model.position.z * 0.60)
                        : Qt.vector3d(model.position.x - 1.85, model.position.y, model.position.z)
                    scale: Qt.vector3d(0.045, 0.045, 0.045)
                    visible: model.active
                    materials: [
                        PrincipledMaterial {
                            baseColor: model.valid ? "#58a6ff" : "#5c6672"
                            emissiveFactor: model.valid ? Qt.vector3d(0.12, 0.19, 0.32) : Qt.vector3d(0, 0, 0)
                            roughness: 0.35
                            metalness: 0.18
                        }
                    ]
                }
            }

            Repeater3D {
                id: rightMarkerRepeater
                model: root.rightMarkers
                delegate: Model {
                    source: "#Sphere"
                    position: root.rigAssetReady
                        ? Qt.vector3d(model.position.x * 0.60 + 1.60, model.position.y * 0.60, model.position.z * 0.60)
                        : Qt.vector3d(model.position.x + 1.85, model.position.y, model.position.z)
                    scale: Qt.vector3d(0.045, 0.045, 0.045)
                    visible: model.active
                    materials: [
                        PrincipledMaterial {
                            baseColor: model.valid ? "#58a6ff" : "#5c6672"
                            emissiveFactor: model.valid ? Qt.vector3d(0.12, 0.19, 0.32) : Qt.vector3d(0, 0, 0)
                            roughness: 0.35
                            metalness: 0.18
                        }
                    ]
                }
            }
        }
    }

    Timer {
        id: rigCacheTimer
        interval: 40
        repeat: false
        onTriggered: root.cacheRigNodes()
    }

    Rectangle {
        visible: !root.rigAssetReady
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        width: Math.min(parent.width - 50, 520)
        height: 64
        radius: 8
        color: "#182332"
        border.color: root.rigAssetFailed ? "#f78166" : "#735f3b"
        border.width: 1
        opacity: 0.94
        Text {
            anchors.fill: parent
            anchors.margins: 12
            text: root.rigAssetFailed
                ? "RIGGED HAND ASSET ERROR\n" + root.rigAssetError
                : (root.rigAssetAvailable
                    ? "Loading persistent rigged LEFT / RIGHT hands…"
                    : (root.surfaceMode
                        ? "Rigged GLB unavailable — MANO diagnostics surface active"
                        : "Rigged hand GLB unavailable — pass --rig-asset"))
            color: root.rigAssetFailed ? "#ffb4a5" : "#f6c78e"
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            wrapMode: Text.Wrap
            elide: Text.ElideRight
        }
    }

    Rectangle {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 16
        width: 116
        height: 48
        radius: 7
        color: "#161f2b"
        border.color: "#344255"
        border.width: 1

        Column {
            anchors.centerIn: parent
            spacing: 2
            Text { text: "LEFT"; color: "#f1f5f9"; font.pixelSize: 15; font.bold: true }
            Text { text: root.leftState; color: root.leftDimmed ? "#f78166" : "#8fa4ba"; font.pixelSize: 10 }
        }
    }

    Rectangle {
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: 16
        width: 124
        height: 48
        radius: 7
        color: "#161f2b"
        border.color: "#344255"
        border.width: 1

        Column {
            anchors.centerIn: parent
            spacing: 2
            Text { text: "RIGHT"; color: "#f1f5f9"; font.pixelSize: 15; font.bold: true }
            Text { text: root.rightState; color: root.rightDimmed ? "#f78166" : "#8fa4ba"; font.pixelSize: 10 }
        }
    }

    Rectangle {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 14
        width: 190
        height: 28
        radius: 6
        color: "#161f2b"
        opacity: 0.92
        Text {
            anchors.centerIn: parent
            text: "Drag to orbit  ·  wheel to zoom"
            color: "#9eacbc"
            font.pixelSize: 11
        }
    }

    MouseArea {
        id: orbitArea
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        hoverEnabled: true
        property real lastX: 0
        property real lastY: 0

        onPressed: {
            lastX = mouse.x
            lastY = mouse.y
        }
        onPositionChanged: {
            if (!pressed)
                return
            root.cameraYaw += (mouse.x - lastX) * 0.34
            root.cameraPitch = Math.max(-35, Math.min(35, root.cameraPitch + (mouse.y - lastY) * 0.24))
            lastX = mouse.x
            lastY = mouse.y
        }
        onWheel: {
            root.cameraDistance = Math.max(5.0, Math.min(14.0, root.cameraDistance - wheel.angleDelta.y / 720.0))
            wheel.accepted = true
        }
    }

    Connections {
        target: appState
        function onRigPoseChanged() {
            root.rigPose = appState.rigPose
            root.applyRigPose(root.rigPose)
        }
        function onResetViewRequested() { root.resetView() }
    }

    Component.onCompleted: {
        if (root.rigAssetAvailable)
            rigCacheTimer.restart()
        root.applyRigPose(root.rigPose)
    }
}
