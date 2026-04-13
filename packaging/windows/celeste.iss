#define MyAppName "Celeste"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Cognameton"
#define MyAppExeName "Celeste.exe"

[Setup]
AppId={{A57E6339-10B6-4B75-9075-4F95F79642A8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=..\..\dist\installer
OutputBaseFilename=Celeste-Setup-{#MyAppVersion}
SetupIconFile=..\..\assets\celeste_icon.ico
Compression=lzma
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=max
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter={#MyAppExeName}
RestartApplications=no
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\..\dist\Celeste\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Celeste"; Flags: nowait postinstall skipifsilent
