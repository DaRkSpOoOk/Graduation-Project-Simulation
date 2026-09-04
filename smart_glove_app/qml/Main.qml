import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import QtQuick.Window 6.8
import "components"

ApplicationWindow {
    id: window
    visible: true
    width: 1600
    height: 1000
    minimumWidth: 1120
    minimumHeight: 760
    title: "Arabic Smart Glove"
    color: "#0d1117"
    property var appState
    property var leftGeometryObject
    property var rightGeometryObject
    property var leftMarkerModel
    property var rightMarkerModel

    // FrameAnimation is synchronized with Qt Quick's rendered frames, so the
    // optional diagnostics value is a presentation FPS measurement rather
    // than a timer-loop estimate.
    FrameAnimation {
        running: true
        onTriggered: appState.recordRenderFrame()
    }

    background: Rectangle {
        color: "#0d1117"
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 22
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 52
            spacing: 16

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 1
                Text {
                    text: "Arabic Smart Glove"
                    color: "#f1f5f9"
                    font.pixelSize: 24
                    font.bold: true
                }
                Text {
                    text: "Core-28 · persistent dual-hand playback"
                    color: "#8fa4ba"
                    font.pixelSize: 12
                }
            }

            Rectangle {
                implicitWidth: topologyLabel.implicitWidth + 24
                implicitHeight: 38
                radius: 7
                color: appState.surfaceMode ? "#20382f" : "#3b3025"
                border.color: appState.surfaceMode ? "#356b57" : "#7b5b38"
                border.width: 1
                Text {
                    id: topologyLabel
                    anchors.centerIn: parent
                    text: appState.surfaceMode ? "MANO surface" : appState.topologyStatus
                    color: appState.surfaceMode ? "#8bd1b2" : "#f6c78e"
                    font.pixelSize: 11
                    font.bold: true
                    elide: Text.ElideRight
                }
            }

            Text {
                text: appState.playbackPlaying ? "Playing" : "Ready"
                color: appState.playbackPlaying ? "#79c0ff" : "#8fa4ba"
                font.pixelSize: 13
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 390
            spacing: 14

            HandViewport {
                id: handViewport
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: 700
                leftGeometry: leftGeometryObject
                rightGeometry: rightGeometryObject
                leftMarkers: leftMarkerModel
                rightMarkers: rightMarkerModel
                surfaceMode: appState.surfaceMode
                leftDimmed: appState.leftDimmed
                rightDimmed: appState.rightDimmed
                leftState: appState.leftState
                rightState: appState.rightState
            }

            RecognitionCard {
                Layout.preferredWidth: 330
                Layout.fillHeight: true
                status: appState.recognitionStatus
                role: appState.recognitionRole
                scope: appState.recognitionScope
                reference: appState.recognitionReference
                checkpoint: appState.recognitionCheckpoint
                expected: appState.expectedCharacter
                predicted: appState.predictedCharacter
                confidence: appState.confidenceText
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 54
            color: "#161b22"
            border.color: "#283443"
            border.width: 1
            radius: 10

            RowLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 12
                LayoutMirroring.enabled: true

                Text {
                    text: "Queued text"
                    color: "#8fa4ba"
                    font.pixelSize: 11
                }
                Text {
                    text: appState.queuedText.length ? appState.queuedText : "—"
                    color: "#f1f5f9"
                    font.pixelSize: 21
                    font.bold: true
                    horizontalAlignment: Text.AlignRight
                    Layout.fillWidth: true
                }
                Text {
                    text: appState.activeLabel
                    color: "#79c0ff"
                    font.pixelSize: 12
                }
                Text {
                    text: appState.statusMessage
                    color: "#8fa4ba"
                    font.pixelSize: 11
                    elide: Text.ElideRight
                    Layout.preferredWidth: 420
                }
            }
        }

            ArabicKeyboard {
                id: arabicKeyboard
                Layout.fillWidth: true
                Layout.preferredHeight: 244
                rows: appState.keyboardRows
            }

            PlaybackControls {
                Layout.fillWidth: true
            Layout.preferredHeight: 62
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 34
            spacing: 12

            TextField {
                id: textInput
                Layout.fillWidth: true
                Layout.preferredHeight: 34
                placeholderText: "Type Arabic Core-28 text, then press Enter"
                horizontalAlignment: Text.AlignRight
                color: "#f1f5f9"
                placeholderTextColor: "#6f8092"
                font.pixelSize: 13
                onAccepted: {
                    appState.enqueueText(text)
                    clear()
                }
                background: Rectangle {
                    radius: 7
                    color: "#161b22"
                    border.color: textInput.activeFocus ? "#58a6ff" : "#283443"
                    border.width: 1
                }
            }

            Button {
                id: queueTextButton
                text: "Queue typed text"
                Layout.preferredWidth: 142
                Layout.preferredHeight: 34
                onClicked: {
                    appState.enqueueText(textInput.text)
                    textInput.clear()
                }
                contentItem: Text {
                    text: queueTextButton.text
                    color: "#0d1117"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    radius: 7
                    color: queueTextButton.hovered ? "#79c0ff" : "#58a6ff"
                }
            }

            ProgressBar {
                id: queueProgress
                Layout.preferredWidth: 150
                Layout.preferredHeight: 6
                value: appState.queueProgress
                background: Rectangle { radius: 3; color: "#26303d" }
                contentItem: Item {
                    Rectangle { width: queueProgress.visualPosition * parent.width; height: parent.height; radius: 3; color: "#58a6ff" }
                }
            }
            Text {
                text: appState.queueCompleted + " / " + appState.queueCount
                color: "#8191a3"
                font.pixelSize: 11
            }
        }

        Rectangle {
            visible: appState.diagnosticsVisible
            Layout.fillWidth: true
            Layout.preferredHeight: 36
            color: "#161b22"
            border.color: "#283443"
            border.width: 1
            radius: 7
            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 22
                Text { text: "Render FPS: " + appState.renderFps.toFixed(1); color: "#c9d1d9"; font.pixelSize: 11 }
                Text { text: "Active source: " + appState.activeSequenceFps.toFixed(1) + " FPS"; color: "#c9d1d9"; font.pixelSize: 11 }
                Text { text: "Frame: " + appState.frameIndex + " / " + appState.frameCount; color: "#c9d1d9"; font.pixelSize: 11 }
                Text { text: "API: " + appState.graphicsApi; color: "#c9d1d9"; font.pixelSize: 11; Layout.fillWidth: true }
                Text { text: appState.topologyStatus; color: "#f6c78e"; font.pixelSize: 10 }
            }
        }
    }

    onClosing: appState.shutdown()
}
