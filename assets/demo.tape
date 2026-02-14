# OMEGA Demo — Cross-Session Memory Recall
# Run: cd ~/Projects/omega-public && vhs assets/demo.tape

Output assets/demo.gif

Require bash

Set Shell "bash"
Set FontSize 17
Set FontFamily "Menlo"
Set Width 860
Set Height 500
Set Padding 24
Set Margin 16
Set BorderRadius 10
Set WindowBar Colorful
Set WindowBarSize 40
Set Framerate 15
Set PlaybackSpeed 1
Set LoopOffset 0
Set CursorBlink false

Set Theme {"name":"omega","background":"#0f1117","foreground":"#e2e8f0","cursor":"#a78bfa","selection":"#2d2b55","black":"#0f1117","red":"#f87171","green":"#4ade80","yellow":"#fbbf24","blue":"#60a5fa","magenta":"#a78bfa","cyan":"#67e8f9","white":"#e2e8f0","brightBlack":"#4a5568","brightRed":"#fca5a5","brightGreen":"#86efac","brightYellow":"#fde68a","brightBlue":"#93c5fd","brightMagenta":"#c4b5fd","brightCyan":"#a5f3fc","brightWhite":"#ffffff"}

Hide
Type "bash assets/demo.sh"
Enter
Sleep 500ms
Show

Sleep 28s
