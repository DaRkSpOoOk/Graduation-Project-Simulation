import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "components"

// TASK-007G application shell.
//
// Layout priority: the hands are the hero element and take the whole upper
// area. Everything technical lives behind the diagnostics drawer.
ApplicationWindow {
    id: window
    visible: true
    width: 1500
    height: 980
    minimumWidth: 1040
    minimumHeight: 720
    title: "Arabic Smart Glove"
    color: "#080a0e"

    property var appState
    property var leftGeometryObject
    property var rightGeometryObject
    property string leftAssetUrl: ""
    property string rightAssetUrl: ""
    property var rigProfile: ({})
    property bool debugManoPoints: false
    property var leftSensorModel
    property var rightSensorModel
    property var sensorLayout: []

    readonly property color accent: "#4f9cf0"
    readonly property color textPrimary: "#e8eef6"
    readonly property color textMuted: "#7d8ea3"
    readonly property color panel: "#11151c"
    readonly property color panelBorder: "#1e2733"

    FrameAnimation {
        running: true
        onTriggered: appState.recordRenderFrame()
    }

    component Stat: ColumnLayout {
        property string title: ""
        property string value: ""
        spacing: 1
        Text { text: parent.title; color: window.textMuted; font.pixelSize: 10; elide: Text.ElideRight }
        Text { text: parent.value; color: "#c3d0de"; font.pixelSize: 12; elide: Text.ElideRight }
    }

    component GhostButton: Rectangle {
        id: ghost
        property string label: ""
        property bool active: false
        signal activated()
        implicitWidth: Math.max(76, ghostText.implicitWidth + 26)
        implicitHeight: 34
        radius: 8
        color: ghost.active ? "#16283d" : (ghostMouse.containsMouse ? "#1a222d" : "#141a23")
        border.color: ghost.active ? window.accent : "#232d3a"
        border.width: 1
        Text {
            id: ghostText
            anchors.centerIn: parent
            text: ghost.label
            color: ghost.active ? window.accent : "#b6c4d4"
            font.pixelSize: 12
            font.bold: ghost.active
        }
        MouseArea {
            id: ghostMouse
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: ghost.activated()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 12

        // ---- title bar -----------------------------------------------------
        RowLayout {
            Layout.fillWidth: true
            spacing: 14

            ColumnLayout {
                spacing: 0
                Text {
                    text: "Arabic Smart Glove"
                    color: window.textPrimary
                    font.pixelSize: 21
                    font.bold: true
                }
                Text {
                    text: "Core-28 dual-hand sign visualisation"
                    color: window.textMuted
                    font.pixelSize: 11
                }
            }

            Item { Layout.fillWidth: true }

            Rectangle {
                visible: appState.recognitionStatus !== "Recognition disabled"
                implicitWidth: recogText.implicitWidth + 22
                implicitHeight: 28
                radius: 14
                color: "#132218"
                border.color: "#2a5540"; border.width: 1
                Text {
                    id: recogText
                    anchors.centerIn: parent
                    text: "Recognition on"
                    color: "#6fd39b"; font.pixelSize: 11; font.bold: true
                }
            }

            Text {
                text: appState.playbackPlaying ? "Playing" : "Ready"
                color: appState.playbackPlaying ? window.accent : window.textMuted
                font.pixelSize: 12
            }

            GhostButton {
                label: appState.viewMode === "PALM" ? "Palm view" : "Back view"
                active: true
                onActivated: appState.toggleViewMode()
            }
            GhostButton {
                label: "Reset"
                onActivated: appState.resetView()
            }
            GhostButton {
                label: "Diagnostics"
                active: appState.diagnosticsVisible
                onActivated: appState.toggleDiagnostics()
            }
            GhostButton {
                label: appState.sensorPanelVisible ? "Close sensors" : "Sensors"
                active: appState.sensorPanelVisible
                onActivated: appState.toggleSensorPanel()
            }
        }

        // ---- hero stage ----------------------------------------------------
        HandStage {
            id: stage
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 340

            leftAssetUrl: window.leftAssetUrl
            rightAssetUrl: window.rightAssetUrl
            assetAvailable: appState.rigAssetAvailable
            rigProfile: window.rigProfile
            handPose: appState.handPose
            viewMode: appState.viewMode
            materialMode: appState.materialMode
            debugManoPoints: window.debugManoPoints
            leftGeometry: window.leftGeometryObject
            rightGeometry: window.rightGeometryObject
            sensorLayout: window.sensorLayout
            sensorValidity: appState.sensorValidity
            sensorsEnabled: appState.sensorsEnabled
            sensorVisibilityMode: appState.sensorVisibilityMode
            selectedSensorId: appState.selectedSensorId
            leftState: appState.leftState
            rightState: appState.rightState
            leftDimmed: appState.leftDimmed
            rightDimmed: appState.rightDimmed
        }

        // ---- recognised text ------------------------------------------------
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 62
            radius: 10
            color: window.panel
            border.color: window.panelBorder
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 18
                spacing: 16

                ColumnLayout {
                    spacing: 1
                    Text { text: "Signing"; color: window.textMuted; font.pixelSize: 10 }
                    Text {
                        text: appState.activeLabel
                        color: window.accent
                        font.pixelSize: 15
                        font.bold: true
                    }
                }

                Item { Layout.fillWidth: true }

                Text {
                    text: appState.queuedText.length ? appState.queuedText : "—"
                    color: window.textPrimary
                    font.pixelSize: 30
                    font.bold: true
                    horizontalAlignment: Text.AlignRight
                }

                Rectangle {
                    Layout.preferredWidth: 96
                    Layout.preferredHeight: 5
                    radius: 3
                    color: "#1c2530"
                    Rectangle {
                        width: parent.width * Math.max(0, Math.min(1, appState.queueProgress))
                        height: parent.height
                        radius: 3
                        color: window.accent
                        Behavior on width { NumberAnimation { duration: 140 } }
                    }
                }
                Text {
                    text: appState.queueCompleted + " / " + appState.queueCount
                    color: window.textMuted
                    font.pixelSize: 11
                }
            }
        }

        // ---- keyboard -------------------------------------------------------
        ArabicKeyboard {
            Layout.fillWidth: true
            Layout.preferredHeight: 178
            rows: appState.keyboardRows
        }

        // ---- controls -------------------------------------------------------
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            GhostButton { label: appState.playbackPlaying ? "Pause" : "Play"; active: true; onActivated: appState.playPause() }
            GhostButton { label: "Restart"; onActivated: appState.restart() }
            GhostButton { label: "Space"; onActivated: appState.appendSpace() }
            GhostButton { label: "Backspace"; onActivated: appState.backspace() }
            GhostButton { label: "Clear"; onActivated: appState.clearQueue() }

            Item { Layout.fillWidth: true }

            TextField {
                id: textInput
                Layout.preferredWidth: 300
                Layout.preferredHeight: 34
                placeholderText: "Type Core-28 text, then press Enter"
                horizontalAlignment: Text.AlignRight
                color: window.textPrimary
                placeholderTextColor: "#5f7085"
                font.pixelSize: 13
                onAccepted: { appState.enqueueText(text); clear() }
                background: Rectangle {
                    radius: 8
                    color: "#141a23"
                    border.color: textInput.activeFocus ? window.accent : "#232d3a"
                    border.width: 1
                }
            }
        }

        // ---- status ---------------------------------------------------------
        Text {
            Layout.fillWidth: true
            text: appState.statusMessage
            color: window.textMuted
            font.pixelSize: 11
            elide: Text.ElideRight
        }

        // ---- diagnostics drawer ---------------------------------------------
        Rectangle {
            visible: appState.diagnosticsVisible
            Layout.fillWidth: true
            Layout.preferredHeight: 156
            radius: 10
            color: window.panel
            border.color: window.panelBorder
            border.width: 1

            GridLayout {
                anchors.fill: parent
                anchors.margins: 14
                columns: 4
                rowSpacing: 6
                columnSpacing: 26

                Stat { title: "Render FPS"; value: appState.renderFps.toFixed(1) }
                Stat { title: "Source FPS"; value: appState.activeSequenceFps.toFixed(1) }
                Stat { title: "Frame"; value: appState.frameIndex + " / " + appState.frameCount }
                Stat { title: "Graphics"; value: appState.graphicsApi }
                Stat { title: "Sample"; value: appState.currentSampleId }
                Stat { title: "Hand asset"; value: appState.rigAssetStatus }
                Stat { title: "Expected"; value: appState.expectedCharacter }
                Stat { title: "Predicted"; value: appState.predictedCharacter + "  " + appState.confidenceText }
                Stat { title: "Recognition"; value: appState.recognitionStatus }
                Stat { title: "Reference"; value: appState.recognitionReference }
                Stat { title: "Left / Right"; value: appState.leftState + " / " + appState.rightState }

                RowLayout {
                    Layout.columnSpan: 4
                    spacing: 6
                    Text { text: "Appearance"; color: window.textMuted; font.pixelSize: 10 }
                    GhostButton { label: "Skin"; active: appState.materialMode === "SKIN"; onActivated: appState.setMaterialMode("SKIN") }
                    GhostButton { label: "Glove"; active: appState.materialMode === "GLOVE"; onActivated: appState.setMaterialMode("GLOVE") }
                    GhostButton { label: "Wire"; active: appState.materialMode === "WIREFRAME"; onActivated: appState.setMaterialMode("WIREFRAME") }

                    Text { text: "Speed"; color: window.textMuted; font.pixelSize: 10; Layout.leftMargin: 14 }
                    GhostButton { label: "0.5x"; active: appState.speed === 0.5; onActivated: appState.setSpeed(0.5) }
                    GhostButton { label: "1x"; active: appState.speed === 1.0; onActivated: appState.setSpeed(1.0) }
                    GhostButton { label: "2x"; active: appState.speed === 2.0; onActivated: appState.setSpeed(2.0) }

                    GhostButton {
                        Layout.leftMargin: 14
                        label: "Smoothing"
                        active: appState.smoothRendering
                        onActivated: appState.setSmoothRendering(!appState.smoothRendering)
                    }
                    GhostButton {
                        label: "Inspect"
                        active: stage.inspectMode
                        onActivated: { stage.inspectMode = !stage.inspectMode; if (!stage.inspectMode) stage.resetView() }
                    }
                    Item { Layout.fillWidth: true }
                }
            }
        }
    }

    SensorPanel {
        id: sensorPanel
        visible: Boolean(appState && appState.sensorPanelVisible)
        anchors.top: parent.top
        anchors.topMargin: 68
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 18
        width: Math.min(370, Math.max(310, parent.width * 0.245))
        z: 50
        appState: window.appState
        leftModel: window.leftSensorModel
        rightModel: window.rightSensorModel
    }

    onClosing: appState.shutdown()
}
