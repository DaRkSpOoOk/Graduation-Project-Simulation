import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    id: root

    property var rows: []
    signal keyPressed(string character)

    color: "#161b22"
    border.color: "#283443"
    border.width: 1
    radius: 10

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 10
        LayoutMirroring.enabled: true

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "Arabic Core-28 keyboard"
                color: "#f1f5f9"
                font.pixelSize: 16
                font.bold: true
                Layout.fillWidth: true
            }
            Text {
                text: "Each click queues one explicit sign event"
                color: "#8191a3"
                font.pixelSize: 11
            }
        }

        Repeater {
            model: root.rows
            delegate: RowLayout {
                property var rowValues: modelData
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                spacing: 8
                layoutDirection: Qt.RightToLeft

                Repeater {
                    model: rowValues
                    delegate: Button {
                        id: keyButton
                        property string character: modelData
                        text: character
                        height: 54
                        Layout.preferredWidth: Math.max(48, (root.width - 44 - (rowValues.length - 1) * 8) / rowValues.length)
                        Layout.fillWidth: true
                        font.pixelSize: 23
                        font.bold: true
                        hoverEnabled: true
                        onClicked: {
                            root.keyPressed(character)
                            appState.enqueueCharacter(character)
                        }

                        contentItem: Text {
                            text: keyButton.text
                            color: keyButton.down ? "#0d1117" : "#f1f5f9"
                            font: keyButton.font
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        background: Rectangle {
                            radius: 7
                            color: keyButton.down ? "#79c0ff" : (keyButton.hovered ? "#26384c" : "#202b38")
                            border.color: keyButton.down ? "#79c0ff" : (keyButton.hovered ? "#58a6ff" : "#344255")
                            border.width: 1
                        }
                    }
                }
            }
        }
    }
}
