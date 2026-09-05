import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    id: root

    property string status: "Recognition disabled"
    property string role: "Visualization-only mode"
    property string scope: "—"
    property string reference: "—"
    property string checkpoint: "—"
    property string expected: "—"
    property string predicted: "—"
    property string confidence: "—"

    color: "#161b22"
    border.color: "#283443"
    border.width: 1
    radius: 10

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 12
        LayoutMirroring.enabled: true

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "Recognition"
                color: "#f1f5f9"
                font.pixelSize: 17
                font.bold: true
                Layout.fillWidth: true
            }
            Rectangle {
                implicitWidth: statusText.implicitWidth + 18
                implicitHeight: 26
                radius: 6
                color: root.status === "Recognition disabled" ? "#222a35" : "#20382f"
                border.color: root.status === "Recognition disabled" ? "#465363" : "#356b57"
                Text {
                    id: statusText
                    anchors.centerIn: parent
                    text: root.status
                    color: root.status === "Recognition disabled" ? "#aab6c4" : "#8bd1b2"
                    font.pixelSize: 10
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#283443"
        }

        Text { text: "Model role"; color: "#8fa4ba"; font.pixelSize: 11 }
        Text {
            text: root.role
            color: "#f1f5f9"
            font.pixelSize: 14
            font.bold: true
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Text { text: "Training scope"; color: "#8fa4ba"; font.pixelSize: 11 }
        Text {
            text: root.scope
            color: "#c9d1d9"
            font.pixelSize: 12
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Text { text: "Scientific reference"; color: "#8fa4ba"; font.pixelSize: 11 }
        Text {
            text: root.reference
            color: "#c9d1d9"
            font.pixelSize: 11
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }

        Text { text: "Checkpoint"; color: "#8fa4ba"; font.pixelSize: 11 }
        Text {
            text: root.checkpoint
            color: "#c9d1d9"
            font.pixelSize: 11
            elide: Text.ElideMiddle
            Layout.fillWidth: true
        }

        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#283443"
        }

        GridLayout {
            columns: 2
            rowSpacing: 10
            columnSpacing: 16
            Layout.fillWidth: true

            Text { text: "Expected"; color: "#8fa4ba"; font.pixelSize: 11 }
            Text { text: root.expected; color: "#f1f5f9"; font.pixelSize: 22; font.bold: true; horizontalAlignment: Text.AlignRight; Layout.fillWidth: true }
            Text { text: "Predicted"; color: "#8fa4ba"; font.pixelSize: 11 }
            Text { text: root.predicted; color: "#79c0ff"; font.pixelSize: 22; font.bold: true; horizontalAlignment: Text.AlignRight; Layout.fillWidth: true }
            Text { text: "Max probability"; color: "#8fa4ba"; font.pixelSize: 11 }
            Text { text: root.confidence; color: "#c9d1d9"; font.pixelSize: 14; horizontalAlignment: Text.AlignRight; Layout.fillWidth: true }
        }

        Item { Layout.fillHeight: true }

        Text {
            text: "Predictions are demo outputs for the selected stored sequence; they are not a new evaluation."
            color: "#6f8092"
            font.pixelSize: 10
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }
    }
}
