import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// TASK-007J sensor drawer.  Numeric values are source-frame readings from
// the frozen TASK-008 virtual-glove contract; the drawer never reads the
// presentation pose or fabricates ADC channels.
Rectangle {
    id: panel

    property var appState
    property var leftModel
    property var rightModel
    readonly property var activeModel: appState && appState.sensorHand === "RIGHT"
        ? panel.rightModel : panel.leftModel

    radius: 14
    color: "#10161f"
    border.color: "#2a3645"
    border.width: 1
    clip: true

    component PanelButton: Rectangle {
        id: button
        property string label: ""
        property bool active: false
        signal activated()
        implicitWidth: Math.max(58, buttonText.implicitWidth + 22)
        implicitHeight: 30
        radius: 8
        color: button.active ? "#1c3448" : (buttonMouse.containsMouse ? "#1b2531" : "#151d27")
        border.color: button.active ? "#4f9cf0" : "#283544"
        border.width: 1
        Text {
            id: buttonText
            anchors.centerIn: parent
            text: button.label
            color: button.active ? "#8ec6ff" : "#b7c5d5"
            font.pixelSize: 11
            font.bold: button.active
        }
        MouseArea {
            id: buttonMouse
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: button.activated()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 9

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "SENSORS"
                color: "#e8eef6"
                font.pixelSize: 15
                font.bold: true
                font.letterSpacing: 1.1
            }
            Item { Layout.fillWidth: true }
            PanelButton {
                label: "×"
                onActivated: panel.appState.setSensorPanelVisible(false)
            }
        }

        Text {
            Layout.fillWidth: true
            text: "20 logical packages · 19 Hall + palm IMU"
            color: "#8292a5"
            font.pixelSize: 10
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            PanelButton {
                Layout.fillWidth: true
                label: "LEFT"
                active: Boolean(panel.appState && panel.appState.sensorHand === "LEFT")
                onActivated: panel.appState.setSensorHand("LEFT")
            }
            PanelButton {
                Layout.fillWidth: true
                label: "RIGHT"
                active: Boolean(panel.appState && panel.appState.sensorHand === "RIGHT")
                onActivated: panel.appState.setSensorHand("RIGHT")
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            PanelButton {
                Layout.fillWidth: true
                label: panel.appState && panel.appState.sensorsEnabled ? "Sensors ON" : "Sensors OFF"
                active: Boolean(panel.appState && panel.appState.sensorsEnabled)
                onActivated: panel.appState.toggleSensors()
            }
            PanelButton {
                Layout.fillWidth: true
                label: panel.appState && panel.appState.sensorVisibilityMode === "OVERLAY" ? "Overlay" : "Physical"
                active: Boolean(panel.appState && panel.appState.sensorVisibilityMode === "OVERLAY")
                onActivated: panel.appState.toggleSensorVisibilityMode()
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: statusText.implicitHeight + 14
            radius: 8
            color: "#141e29"
            border.color: "#243344"
            border.width: 1
            Text {
                id: statusText
                anchors.fill: parent
                anchors.margins: 7
                text: panel.appState ? panel.appState.sensorStatus : ""
                color: panel.appState && panel.appState.sensorStatus.indexOf("TRANSITION") === 0 ? "#f0c77a" : "#9eb3c9"
                font.pixelSize: 10
                wrapMode: Text.Wrap
                verticalAlignment: Text.AlignVCenter
            }
        }

        ListView {
            id: sensorList
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 5
            model: panel.activeModel
            section.property: "group"
            section.criteria: ViewSection.FullString
            section.delegate: Rectangle {
                width: sensorList.width
                height: 22
                color: "transparent"
                Text {
                    anchors.left: parent.left
                    anchors.leftMargin: 2
                    anchors.bottom: parent.bottom
                    text: section
                    color: "#6e849b"
                    font.pixelSize: 10
                    font.bold: true
                    font.letterSpacing: 1.2
                }
            }

            delegate: Rectangle {
                width: sensorList.width - 6
                height: hasAngle ? 56 : 77
                radius: 8
                color: selected ? "#243b4c" : "#151d27"
                border.color: selected ? "#e7b866" : "#202c3a"
                border.width: 1

                ColumnLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 10
                    anchors.rightMargin: 10
                    anchors.topMargin: 7
                    anchors.bottomMargin: 6
                    spacing: 2

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            Layout.fillWidth: true
                            text: displayName
                            color: "#dce6f1"
                            font.pixelSize: 11
                            font.bold: true
                            elide: Text.ElideRight
                        }
                        Text {
                            text: shortId
                            color: selected ? "#ffd98f" : "#7890a7"
                            font.pixelSize: 10
                            font.bold: true
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        visible: hasAngle
                        Text { text: "Normalized"; color: "#7d8ea3"; font.pixelSize: 9 }
                        Text { text: normalizedText; color: "#c7d6e6"; font.pixelSize: 10; font.family: "Consolas" }
                        Item { Layout.fillWidth: true }
                        Text { text: "Derived angle"; color: "#7d8ea3"; font.pixelSize: 9 }
                        Text { text: angleText; color: "#c7d6e6"; font.pixelSize: 10; font.family: "Consolas" }
                    }
                    Text {
                        Layout.fillWidth: true
                        visible: !hasAngle
                        text: quaternionText
                        color: "#c7d6e6"
                        font.pixelSize: 10
                        font.family: "Consolas"
                        lineHeight: 1.0
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "Valid"; color: "#7d8ea3"; font.pixelSize: 9 }
                        Text {
                            text: validText
                            color: valid ? "#73d29d" : "#e19a7d"
                            font.pixelSize: 9
                            font.bold: true
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            visible: !hasAngle
                            text: "WXYZ source order"
                            color: "#63758a"
                            font.pixelSize: 9
                        }
                    }
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: panel.appState.selectSensor(sensorId)
                }
            }
        }

        Text {
            Layout.fillWidth: true
            text: "Values: frozen TASK-008 source frame · no ADC channels"
            color: "#607287"
            font.pixelSize: 9
            elide: Text.ElideRight
        }
    }
}
