import QtQuick

// Readable H/IMU badge used by the optional XRAY presentation.  It is a 2D
// label projected from a 3D marker node, so it remains visible through the
// hand without moving the physical anchor to an anatomically false surface.
Item {
    id: root

    property var projector
    property var markerNode
    property var definition: ({})
    property string handSide: ""
    property bool sensorsEnabled: false
    property bool sourceValid: false
    property bool selected: false
    property int revision: 0

    readonly property bool isImu: Boolean(
        root.definition && root.definition.marker === "IMU"
    )
    readonly property real badgeWidth: root.isImu ? 38 : 20
    readonly property real badgeHeight: root.isImu ? 20 : 20

    function projected() {
        root.revision
        if (!root.projector || !root.markerNode)
            return Qt.vector3d(-1000, -1000, -1)
        // RuntimeLoader node scenePosition can be one render tick behind when
        // a pose is first bound.  Mapping the marker's current local position
        // through its owning hand gives the same final-rig transform without
        // introducing a second landmark-based position path.
        var hand = root.markerNode.handNode
        if (!hand)
            return Qt.vector3d(-1000, -1000, -1)
        var scenePosition = hand.mapPositionToScene(root.markerNode.position)
        return root.projector.projectSensorScenePosition(scenePosition)
    }

    width: root.badgeWidth
    height: root.badgeHeight
    x: root.projected().x - width * 0.5
    y: root.projected().y - height * 0.5
    visible: root.sensorsEnabled && Boolean(root.markerNode && root.markerNode.visible)
    z: 30

    Rectangle {
        anchors.fill: parent
        radius: height * 0.5
        color: root.selected ? "#f3b562" : (root.sourceValid ? "#1e8f9a" : "#465260")
        border.color: root.selected ? "#ffe1aa" : "#a8f2ec"
        border.width: 1
        opacity: root.sourceValid || root.selected ? 0.96 : 0.60
    }
    Text {
        anchors.centerIn: parent
        text: root.definition.marker || "H"
        color: root.selected ? "#10151b" : "#f2fbff"
        font.pixelSize: root.isImu ? 9 : 11
        font.bold: true
        font.letterSpacing: root.isImu ? 0.4 : 0
    }
    MouseArea {
        anchors.fill: parent
        enabled: Boolean(root.definition && root.definition.sensorId)
        cursorShape: Qt.PointingHandCursor
        onClicked: {
            if (appState) {
                appState.setSensorHand(root.handSide)
                appState.selectSensor(root.definition.sensorId)
            }
        }
    }
}
