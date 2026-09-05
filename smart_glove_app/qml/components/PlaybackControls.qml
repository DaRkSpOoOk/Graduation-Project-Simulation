import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    id: root

    color: "#161b22"
    border.color: "#283443"
    border.width: 1
    radius: 10

    RowLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        Button {
            id: playButton
            text: appState.playbackPlaying ? "Pause" : "Play"
            onClicked: appState.playPause()
            Layout.preferredWidth: 86
            Layout.preferredHeight: 38
            contentItem: Text { text: playButton.text; color: "#0d1117"; font.bold: true; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            background: Rectangle { radius: 7; color: playButton.hovered ? "#79c0ff" : "#58a6ff" }
        }

        Button {
            id: restartButton
            text: "Restart"
            onClicked: appState.restart()
            Layout.preferredHeight: 38
            contentItem: Text { text: restartButton.text; color: "#d8e2ee"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            background: Rectangle { radius: 7; color: restartButton.hovered ? "#26384c" : "#202b38"; border.color: "#344255"; border.width: 1 }
        }

        Button {
            id: clearButton
            text: "Clear"
            onClicked: appState.clearQueue()
            Layout.preferredHeight: 38
            contentItem: Text { text: clearButton.text; color: "#d8e2ee"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            background: Rectangle { radius: 7; color: clearButton.hovered ? "#3b2a2c" : "#26252b"; border.color: "#4a3b40"; border.width: 1 }
        }

        Button {
            id: backspaceButton
            text: "Backspace"
            onClicked: appState.backspace()
            Layout.preferredHeight: 38
            contentItem: Text { text: backspaceButton.text; color: "#d8e2ee"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            background: Rectangle { radius: 7; color: backspaceButton.hovered ? "#26384c" : "#202b38"; border.color: "#344255"; border.width: 1 }
        }

        Button {
            id: spaceButton
            text: "Space"
            onClicked: appState.appendSpace()
            Layout.preferredHeight: 38
            contentItem: Text { text: spaceButton.text; color: "#d8e2ee"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            background: Rectangle { radius: 7; color: spaceButton.hovered ? "#26384c" : "#202b38"; border.color: "#344255"; border.width: 1 }
        }

        Rectangle { Layout.fillWidth: true; color: "transparent" }

        Text { text: "Mode"; color: "#8191a3"; font.pixelSize: 11 }
        ComboBox {
            id: modeBox
            model: ["Canonical", "Signer 01", "Signer 02", "Signer 03", "Seeded random"]
            currentIndex: appState.exemplarMode === "canonical" ? 0 : (appState.exemplarMode === "signer01" ? 1 : (appState.exemplarMode === "signer02" ? 2 : (appState.exemplarMode === "signer03" ? 3 : 4)))
            onActivated: appState.setExemplarMode(["canonical", "signer01", "signer02", "signer03", "random"][currentIndex])
            Layout.preferredWidth: 128
        }

        Text { text: "Speed"; color: "#8191a3"; font.pixelSize: 11 }
        ComboBox {
            id: speedBox
            model: ["0.5×", "1×", "2×"]
            currentIndex: 1
            onActivated: appState.setSpeed([0.5, 1.0, 2.0][currentIndex])
            Layout.preferredWidth: 74
        }

        CheckBox {
            id: smoothBox
            text: "Smooth rendering"
            checked: appState.smoothRendering
            onToggled: appState.setSmoothRendering(checked)
        }

        Button {
            id: resetButton
            text: "Reset view"
            onClicked: appState.resetView()
            Layout.preferredHeight: 38
            contentItem: Text { text: resetButton.text; color: "#d8e2ee"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            background: Rectangle { radius: 7; color: resetButton.hovered ? "#26384c" : "#202b38"; border.color: "#344255"; border.width: 1 }
        }

        Button {
            id: diagnosticsButton
            text: "Diagnostics"
            onClicked: appState.toggleDiagnostics()
            Layout.preferredHeight: 38
            contentItem: Text { text: diagnosticsButton.text; color: "#d8e2ee"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            background: Rectangle { radius: 7; color: diagnosticsButton.hovered ? "#26384c" : "#202b38"; border.color: "#344255"; border.width: 1 }
        }
    }
}
