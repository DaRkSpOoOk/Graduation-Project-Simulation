import QtQuick
import QtQuick.Layouts

// Core-28 keyboard. The mapping is authoritative and comes from the controller;
// this file only lays it out. One click enqueues exactly one sign event, and a
// repeated letter stays a separate event.
Item {
    id: root
    property var rows: []
    property real keyHeight: 46

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        Repeater {
            model: root.rows
            delegate: RowLayout {
                required property var modelData
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 8
                // Arabic reads right-to-left, so the first character of each row
                // must sit on the right.
                layoutDirection: Qt.RightToLeft

                Repeater {
                    model: parent.modelData
                    delegate: Rectangle {
                        required property string modelData
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: root.keyHeight
                        radius: 9
                        color: keyMouse.pressed ? "#2d6cc4"
                             : keyMouse.containsMouse ? "#1e2733" : "#161c25"
                        border.color: keyMouse.containsMouse ? "#3d5a80" : "#242e3b"
                        border.width: 1

                        Behavior on color { ColorAnimation { duration: 90 } }

                        Text {
                            anchors.centerIn: parent
                            text: parent.modelData
                            color: keyMouse.pressed ? "#ffffff" : "#e3ebf5"
                            font.pixelSize: 25
                            font.family: "Segoe UI"
                        }

                        MouseArea {
                            id: keyMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: appState.enqueueCharacter(parent.modelData)
                        }
                    }
                }
            }
        }
    }
}
