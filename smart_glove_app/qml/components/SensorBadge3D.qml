import QtQuick
import QtQuick3D

// Persistent presentation-only sensor badge.  The node stays in the hand's
// coordinate space; its position is recomputed from the final displayed rig
// whenever the controller publishes a pose.  It never reads raw landmarks.
Node {
    id: root

    property var handNode
    property var definition: ({})
    property var validity: ({})
    property bool sensorsEnabled: false
    property string visibilityMode: "OVERLAY"
    property string selectedSensorId: ""
    property int revision: 0
    property var hallMaterial
    property var imuMaterial
    property var selectedMaterial
    property var invalidMaterial

    readonly property bool sourceValid: Boolean(
        root.validity && root.definition && root.validity[root.definition.sensorId]
    )
    readonly property bool selected: Boolean(
        root.definition && root.definition.sensorId === root.selectedSensorId
    )

    function vectorFrom(values) {
        if (!values || values.length !== 3)
            return Qt.vector3d(0, 0, 0)
        return Qt.vector3d(Number(values[0]), Number(values[1]), Number(values[2]))
    }

    function pointFromBone(bone, localOffset) {
        if (!bone || !root.handNode)
            return Qt.vector3d(0, 0, 0)
        var scenePoint = bone.mapPositionToScene(vectorFrom(localOffset))
        return root.handNode.mapPositionFromScene(scenePoint)
    }

    function resolvedPosition() {
        var definition = root.definition
        var hand = root.handNode
        if (!hand || !hand.ready || !definition)
            return Qt.vector3d(0, 0, 0)

        if (definition.anchorKind === "bone") {
            var bone = hand._bones[definition.anchorBone]
            return pointFromBone(bone, definition.localOffset)
        }

        if (definition.anchorKind === "spread_midpoint") {
            var names = definition.anchorBones || []
            if (names.length !== 2)
                return Qt.vector3d(0, 0, 0)
            var firstBone = hand._bones[names[0]]
            var secondBone = hand._bones[names[1]]
            var palmBone = hand._bones["palm"]
            if (!firstBone || !secondBone || !palmBone)
                return Qt.vector3d(0, 0, 0)
            var first = hand.mapPositionFromScene(firstBone.scenePosition)
            var second = hand.mapPositionFromScene(secondBone.scenePosition)
            var palm = hand.mapPositionFromScene(palmBone.scenePosition)
            var blend = Math.max(0.0, Math.min(1.0, Number(definition.palmBlend || 0.0)))
            var midpoint = Qt.vector3d(
                (first.x + second.x) * 0.5,
                (first.y + second.y) * 0.5,
                (first.z + second.z) * 0.5
            )
            var anchor = Qt.vector3d(
                midpoint.x * (1.0 - blend) + palm.x * blend,
                midpoint.y * (1.0 - blend) + palm.y * blend,
                midpoint.z * (1.0 - blend) + palm.z * blend
            )
            var offset = vectorFrom(definition.localOffset)
            return Qt.vector3d(anchor.x + offset.x, anchor.y + offset.y, anchor.z + offset.z)
        }
        return Qt.vector3d(0, 0, 0)
    }

    position: {
        // Explicit dependency keeps this binding live even though the
        // RuntimeLoader bones are resolved dynamically.
        root.revision
        return root.resolvedPosition()
    }
    visible: root.sensorsEnabled && Boolean(root.handNode && root.handNode.ready)

    Model {
        id: badge
        source: "#Cylinder"
        scale: root.definition && root.definition.marker === "IMU"
            // Qt Quick 3D's built-in primitives are 100-unit meshes; these
            // scales keep a badge at roughly 3–7 cm on the 1.188-unit hand.
            ? Qt.vector3d(0.00070, 0.00010, 0.00035)
            : Qt.vector3d(0.00034, 0.00008, 0.00034)
        eulerRotation: Qt.vector3d(90, 0, 0)
        visible: root.visible
        castsShadows: false
        receivesShadows: false
        pickable: false
        // The 2D overlay supplies the readable badge in OVERLAY mode.  This
        // bias also keeps the 3D badge visible when it is close to the mesh;
        // PHYSICAL mode leaves normal depth testing intact.
        depthBias: root.visibilityMode === "OVERLAY" ? -1.0 : 0.0
        materials: [
            root.selected ? root.selectedMaterial
                          : root.definition && root.definition.marker === "IMU"
                            ? root.imuMaterial
                            : root.sourceValid ? root.hallMaterial : root.invalidMaterial
        ]
    }

    Component.onCompleted: {
        if (root.handNode && root.definition && root.definition.sensorId)
            root.handNode.registerSensorMarker(root.definition.sensorId, root)
    }
    onDefinitionChanged: {
        if (root.handNode && root.definition && root.definition.sensorId)
            root.handNode.registerSensorMarker(root.definition.sensorId, root)
    }
}
