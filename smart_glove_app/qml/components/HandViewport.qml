import QtQuick 6.8
import QtQuick3D 6.8

Item {
    id: root

    property var leftGeometry
    property var rightGeometry
    property var leftMarkers
    property var rightMarkers
    property bool surfaceMode: false
    property bool leftDimmed: false
    property bool rightDimmed: false
    property string leftState: "IDLE"
    property string rightState: "IDLE"
    property real cameraDistance: 8.0
    property real cameraYaw: 0.0
    property real cameraPitch: -4.0

    function resetView() {
        cameraDistance = 8.0
        cameraYaw = 0.0
        cameraPitch = -4.0
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

            Model {
                id: leftSurface
                visible: root.surfaceMode
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
                visible: root.surfaceMode
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

            Model {
                id: leftPointCloud
                visible: !root.surfaceMode
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
                visible: !root.surfaceMode
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
                    position: Qt.vector3d(model.position.x - 1.85, model.position.y, model.position.z)
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
                    position: Qt.vector3d(model.position.x + 1.85, model.position.y, model.position.z)
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
        function onResetViewRequested() { root.resetView() }
    }
}
