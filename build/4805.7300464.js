"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[4805],{64805:(e,t,n)=>{n.r(t),n.d(t,{Commands:()=>S,default:()=>I,tabSpaceStatus:()=>y});var o=n(80476),a=n(12120),r=n(20487),d=n(53782),i=n(3620),c=n(27485),l=n(92270),s=n(67272),u=n(22743),m=n(87648),g=n(55331),f=n(28877),p=n(82930),C=n(68465);const b="notebook:toggle-autoclosing-brackets",v="console:toggle-autoclosing-brackets";var h;!function(e){e.createNew="fileeditor:create-new",e.createNewMarkdown="fileeditor:create-new-markdown-file",e.changeFontSize="fileeditor:change-font-size",e.lineNumbers="fileeditor:toggle-line-numbers",e.lineWrap="fileeditor:toggle-line-wrap",e.changeTabs="fileeditor:change-tabs",e.matchBrackets="fileeditor:toggle-match-brackets",e.autoClosingBrackets="fileeditor:toggle-autoclosing-brackets",e.autoClosingBracketsUniversal="fileeditor:toggle-autoclosing-brackets-universal",e.createConsole="fileeditor:create-console",e.replaceSelection="fileeditor:replace-selection",e.runCode="fileeditor:run-code",e.runAllCode="fileeditor:run-all",e.markdownPreview="fileeditor:markdown-preview",e.undo="fileeditor:undo",e.redo="fileeditor:redo",e.cut="fileeditor:cut",e.copy="fileeditor:copy",e.paste="fileeditor:paste",e.selectAll="fileeditor:select-all"}(h||(h={}));const _="Editor",w=["autoClosingBrackets","codeFolding","cursorBlinkRate","fontFamily","fontSize","insertSpaces","lineHeight","lineNumbers","lineWrap","matchBrackets","readOnly","rulers","showTrailingSpace","tabSize","wordWrapColumn"];function k(e){const t=Object.assign({},e);for(let t of Object.keys(e))w.includes(t)||delete e[t];return t}let x=k(r.CodeEditor.defaultConfig);var S;!function(e){function t(e){return async function(t,n){var o;const a=n||{},r=await e.execute("console:create",{activate:a.activate,name:null===(o=t.context.contentsModel)||void 0===o?void 0:o.name,path:t.context.path,preferredLanguage:t.context.model.defaultKernelLanguage,ref:t.id,insertMode:"split-bottom"});t.context.pathChanged.connect(((e,n)=>{var o;r.session.setPath(n),r.session.setName(null===(o=t.context.contentsModel)||void 0===o?void 0:o.name)}))}}function n(e){e.editor.setOptions(Object.assign({},x))}function o(e,t,n,o){e.addCommand(h.changeFontSize,{execute:e=>{const n=Number(e.delta);if(Number.isNaN(n))return void console.error(`${h.changeFontSize}: delta arg must be a number`);const a=window.getComputedStyle(document.documentElement),r=parseInt(a.getPropertyValue("--jp-code-font-size"),10),d=x.fontSize||r;return x.fontSize=d+n,t.set(o,"editorConfig",x).catch((e=>{console.error(`Failed to set ${o}: ${e.message}`)}))},label:e=>{var t;return(null!==(t=e.delta)&&void 0!==t?t:0)>0?e.isMenu?n.__("Increase Text Editor Font Size"):n.__("Increase Font Size"):e.isMenu?n.__("Decrease Text Editor Font Size"):n.__("Decrease Font Size")}})}function d(e,t,n,o,a){e.addCommand(h.lineNumbers,{execute:()=>(x.lineNumbers=!x.lineNumbers,t.set(o,"editorConfig",x).catch((e=>{console.error(`Failed to set ${o}: ${e.message}`)}))),isEnabled:a,isToggled:()=>x.lineNumbers,label:n.__("Line Numbers")})}function i(e,t,n,o,a){e.addCommand(h.lineWrap,{execute:e=>(x.lineWrap=e.mode||"off",t.set(o,"editorConfig",x).catch((e=>{console.error(`Failed to set ${o}: ${e.message}`)}))),isEnabled:a,isToggled:e=>{const t=e.mode||"off";return x.lineWrap===t},label:n.__("Word Wrap")})}function c(e,t,n,o){e.addCommand(h.changeTabs,{label:e=>{var t;return e.insertSpaces?n._n("Spaces: %1","Spaces: %1",null!==(t=e.size)&&void 0!==t?t:0):n.__("Indent with Tab")},execute:e=>(x.tabSize=e.size||4,x.insertSpaces=!!e.insertSpaces,t.set(o,"editorConfig",x).catch((e=>{console.error(`Failed to set ${o}: ${e.message}`)}))),isToggled:e=>{const t=!!e.insertSpaces,n=e.size||4;return x.insertSpaces===t&&x.tabSize===n}})}function l(e,t,n,o,a){e.addCommand(h.matchBrackets,{execute:()=>(x.matchBrackets=!x.matchBrackets,t.set(o,"editorConfig",x).catch((e=>{console.error(`Failed to set ${o}: ${e.message}`)}))),label:n.__("Match Brackets"),isEnabled:a,isToggled:()=>x.matchBrackets})}function s(e,t,n,o){e.addCommand(h.autoClosingBrackets,{execute:e=>{var n;return x.autoClosingBrackets=!!(null!==(n=e.force)&&void 0!==n?n:!x.autoClosingBrackets),t.set(o,"editorConfig",x).catch((e=>{console.error(`Failed to set ${o}: ${e.message}`)}))},label:n.__("Auto Close Brackets for Text Editor"),isToggled:()=>x.autoClosingBrackets}),e.addCommand(h.autoClosingBracketsUniversal,{execute:()=>{e.isToggled(h.autoClosingBrackets)||e.isToggled(b)||e.isToggled(v)?(e.execute(h.autoClosingBrackets,{force:!1}),e.execute(b,{force:!1}),e.execute(v,{force:!1})):(e.execute(h.autoClosingBrackets,{force:!0}),e.execute(b,{force:!0}),e.execute(v,{force:!0}))},label:n.__("Auto Close Brackets"),isToggled:()=>e.isToggled(h.autoClosingBrackets)||e.isToggled(b)||e.isToggled(v)})}function u(e,t,n,o){e.addCommand(h.replaceSelection,{execute:e=>{var n,o;const a=e.text||"",r=t.currentWidget;r&&(null===(o=(n=r.content.editor).replaceSelection)||void 0===o||o.call(n,a))},isEnabled:o,label:n.__("Replace Selection in Editor")})}function m(e,n,o,a){e.addCommand(h.createConsole,{execute:o=>{const a=n.currentWidget;if(a)return t(e)(a,o)},isEnabled:a,icon:C.consoleIcon,label:o.__("Create Console for Editor")})}function g(e,t,n,o){e.addCommand(h.runCode,{execute:()=>{var n;const o=null===(n=t.currentWidget)||void 0===n?void 0:n.content;if(!o)return;let a="";const r=o.editor,d=o.context.path,i=p.PathExt.extname(d),c=r.getSelection(),{start:l,end:s}=c;let u=l.column!==s.column||l.line!==s.line;if(u){const e=r.getOffsetAt(c.start),t=r.getOffsetAt(c.end);a=r.model.value.text.substring(e,t)}else if(p.MarkdownCodeBlocks.isMarkdown(i)){const{text:e}=r.model.value,t=p.MarkdownCodeBlocks.findMarkdownCodeBlocks(e);for(const e of t)if(e.startLine<=l.line&&l.line<=e.endLine){a=e.code,u=!0;break}}if(!u){a=r.getLine(c.start.line);const e=r.getCursorPosition();if(e.line+1===r.lineCount){const e=r.model.value.text;r.model.value.text=e+"\n"}r.setCursorPosition({line:e.line+1,column:e.column})}return a?e.execute("console:inject",{activate:!1,code:a,path:d}):Promise.resolve(void 0)},isEnabled:o,label:n.__("Run Code")})}function f(e,t,n,o){e.addCommand(h.runAllCode,{execute:()=>{var n;const o=null===(n=t.currentWidget)||void 0===n?void 0:n.content;if(!o)return;let a="";const r=o.editor.model.value.text,d=o.context.path,i=p.PathExt.extname(d);if(p.MarkdownCodeBlocks.isMarkdown(i)){const e=p.MarkdownCodeBlocks.findMarkdownCodeBlocks(r);for(const t of e)a+=t.code}else a=r;return a?e.execute("console:inject",{activate:!1,code:a,path:d}):Promise.resolve(void 0)},isEnabled:o,label:n.__("Run All Code")})}function w(e,t,n){e.addCommand(h.markdownPreview,{execute:()=>{const n=t.currentWidget;if(!n)return;const o=n.context.path;return e.execute("markdownviewer:open",{path:o,options:{mode:"split-right"}})},isVisible:()=>{const e=t.currentWidget;return e&&".md"===p.PathExt.extname(e.context.path)||!1},icon:C.markdownIcon,label:n.__("Show Markdown Preview")})}function S(e,t,n,o){e.addCommand(h.undo,{execute:()=>{var e;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;n&&n.editor.undo()},isEnabled:()=>{var e;return!!o()&&!!(null===(e=t.currentWidget)||void 0===e?void 0:e.content)},icon:C.undoIcon.bindprops({stylesheet:"menuItem"}),label:n.__("Undo")})}function T(e,t,n,o){e.addCommand(h.redo,{execute:()=>{var e;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;n&&n.editor.redo()},isEnabled:()=>{var e;return!!o()&&!!(null===(e=t.currentWidget)||void 0===e?void 0:e.content)},icon:C.redoIcon.bindprops({stylesheet:"menuItem"}),label:n.__("Redo")})}function y(e,t,n,o){e.addCommand(h.cut,{execute:()=>{var e;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;if(!n)return;const o=n.editor,r=N(o);a.Clipboard.copyToSystem(r),o.replaceSelection&&o.replaceSelection("")},isEnabled:()=>{var e;if(!o())return!1;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;return!!n&&W(n.editor)},icon:C.cutIcon.bindprops({stylesheet:"menuItem"}),label:n.__("Cut")})}function I(e,t,n,o){e.addCommand(h.copy,{execute:()=>{var e;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;if(!n)return;const o=N(n.editor);a.Clipboard.copyToSystem(o)},isEnabled:()=>{var e;if(!o())return!1;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;return!!n&&W(n.editor)},icon:C.copyIcon.bindprops({stylesheet:"menuItem"}),label:n.__("Copy")})}function E(e,t,n,o){e.addCommand(h.paste,{execute:async()=>{var e;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;if(!n)return;const o=n.editor,a=window.navigator.clipboard,r=await a.readText();r&&o.replaceSelection&&o.replaceSelection(r)},isEnabled:()=>{var e;return Boolean(o()&&(null===(e=t.currentWidget)||void 0===e?void 0:e.content))},icon:C.pasteIcon.bindprops({stylesheet:"menuItem"}),label:n.__("Paste")})}function M(e,t,n,o){e.addCommand(h.selectAll,{execute:()=>{var e;const n=null===(e=t.currentWidget)||void 0===e?void 0:e.content;n&&n.editor.execCommand("selectAll")},isEnabled:()=>{var e;return Boolean(o()&&(null===(e=t.currentWidget)||void 0===e?void 0:e.content))},label:n.__("Select All")})}function W(e){const t=e.getSelection(),{start:n,end:o}=t;return n.column!==o.column||n.line!==o.line}function N(e){const t=e.getSelection(),n=e.getOffsetAt(t.start),o=e.getOffsetAt(t.end);return e.model.value.text.substring(n,o)}function B(e,t,n="txt"){return e.execute("docmanager:new-untitled",{path:t,type:"file",ext:n}).then((t=>{if(null!=t)return e.execute("docmanager:open",{path:t.path,factory:_})}))}function P(e,t,n){e.addCommand(h.createNew,{label:e=>{var t,o;return e.isPalette?null!==(t=e.paletteLabel)&&void 0!==t?t:n.__("New Text File"):null!==(o=e.launcherLabel)&&void 0!==o?o:n.__("Text File")},caption:e=>{var t;return null!==(t=e.caption)&&void 0!==t?t:n.__("Create a new text file")},icon:e=>{var t;return e.isPalette?void 0:C.LabIcon.resolve({icon:null!==(t=e.iconName)&&void 0!==t?t:C.textEditorIcon})},execute:n=>{var o;const a=n.cwd||t.defaultBrowser.model.path;return B(e,a,null!==(o=n.fileExt)&&void 0!==o?o:"txt")}})}function F(e,t,n){e.addCommand(h.createNewMarkdown,{label:e=>e.isPalette?n.__("New Markdown File"):n.__("Markdown File"),caption:n.__("Create a new markdown file"),icon:e=>e.isPalette?void 0:C.markdownIcon,execute:n=>{const o=n.cwd||t.defaultBrowser.model.path;return B(e,o,"md")}})}function L(e,t){e.add({command:h.createNew,category:t.__("Other"),rank:1})}function O(e,t){e.add({command:h.createNewMarkdown,category:t.__("Other"),rank:2})}function R(e,t){const n=t.__("Text Editor"),o=h.changeTabs;e.addItem({command:o,args:{insertSpaces:!1,size:4},category:n});for(const t of[1,2,4,8]){const a={insertSpaces:!0,size:t};e.addItem({command:o,args:a,category:n})}}function A(e,t){const n=t.__("Text Editor");e.addItem({command:h.createNew,args:{isPalette:!0},category:n})}function z(e,t){const n=t.__("Text Editor");e.addItem({command:h.createNewMarkdown,args:{isPalette:!0},category:n})}function j(e,t){const n=t.__("Text Editor"),o=h.changeFontSize;let a={delta:1};e.addItem({command:o,args:a,category:n}),a={delta:-1},e.addItem({command:o,args:a,category:n})}function $(e,t){e.editMenu.undoers.add({tracker:t,undo:e=>{e.content.editor.undo()},redo:e=>{e.content.editor.redo()}})}function U(e,t){e.viewMenu.editorViewers.add({tracker:t,toggleLineNumbers:e=>{const t=!e.content.editor.getOption("lineNumbers");e.content.editor.setOption("lineNumbers",t)},toggleWordWrap:e=>{const t="off"===e.content.editor.getOption("lineWrap")?"on":"off";e.content.editor.setOption("lineWrap",t)},toggleMatchBrackets:e=>{const t=!e.content.editor.getOption("matchBrackets");e.content.editor.setOption("matchBrackets",t)},lineNumbersToggled:e=>e.content.editor.getOption("lineNumbers"),wordWrapToggled:e=>"off"!==e.content.editor.getOption("lineWrap"),matchBracketsToggled:e=>e.content.editor.getOption("matchBrackets")})}function K(e,n,o,a){const r=t(n);e.fileMenu.consoleCreators.add({tracker:o,createConsoleLabel:e=>a.__("Create Console for Editor"),createConsole:r})}function J(e,t,n,o,r,d){e.runMenu.codeRunners.add({tracker:n,runLabel:e=>r.__("Run Code"),runAllLabel:e=>r.__("Run All Code"),restartAndRunAllLabel:e=>r.__("Restart Kernel and Run All Code"),isEnabled:e=>!!o.find((t=>{var n;return(null===(n=t.sessionContext.session)||void 0===n?void 0:n.path)===e.context.path})),run:()=>t.execute(h.runCode),runAll:()=>t.execute(h.runAllCode),restartAndRunAll:e=>{const n=o.find((t=>{var n;return(null===(n=t.sessionContext.session)||void 0===n?void 0:n.path)===e.context.path}));if(n)return(d||a.sessionContextDialogs).restart(n.sessionContext).then((e=>(e&&t.execute(h.runAllCode),e)))}})}e.updateSettings=function(e,t){x=k(Object.assign(Object.assign({},r.CodeEditor.defaultConfig),e.get("editorConfig").composite)),t.notifyCommandChanged()},e.updateTracker=function(e){e.forEach((e=>{n(e.content)}))},e.updateWidget=n,e.addCommands=function(e,t,n,a,r,p,C){o(e,t,n,a),d(e,t,n,a,r),i(e,t,n,a,r),c(e,t,n,a),l(e,t,n,a,r),s(e,t,n,a),u(e,p,n,r),m(e,p,n,r),g(e,p,n,r),f(e,p,n,r),w(e,p,n),P(e,C,n),F(e,C,n),S(e,p,n,r),T(e,p,n,r),y(e,p,n,r),I(e,p,n,r),E(e,p,n,r),M(e,p,n,r)},e.addChangeFontSizeCommand=o,e.addLineNumbersCommand=d,e.addWordWrapCommand=i,e.addChangeTabsCommand=c,e.addMatchBracketsCommand=l,e.addAutoClosingBracketsCommand=s,e.addReplaceSelectionCommand=u,e.addCreateConsoleCommand=m,e.addRunCodeCommand=g,e.addRunAllCodeCommand=f,e.addMarkdownPreviewCommand=w,e.addUndoCommand=S,e.addRedoCommand=T,e.addCutCommand=y,e.addCopyCommand=I,e.addPasteCommand=E,e.addSelectAllCommand=M,e.addCreateNewCommand=P,e.addCreateNewMarkdownCommand=F,e.addLauncherItems=function(e,t){L(e,t),O(e,t)},e.addCreateNewToLauncher=L,e.addCreateNewMarkdownToLauncher=O,e.addKernelLanguageLauncherItems=function(e,t,n){for(let o of n)e.add({command:h.createNew,category:t.__("Other"),rank:3,args:o})},e.addPaletteItems=function(e,t){R(e,t),A(e,t),z(e,t),j(e,t)},e.addChangeTabsCommandsToPalette=R,e.addCreateNewCommandToPalette=A,e.addCreateNewMarkdownCommandToPalette=z,e.addChangeFontSizeCommandsToPalette=j,e.addKernelLanguagePaletteItems=function(e,t,n){const o=t.__("Text Editor");for(let t of n)e.addItem({command:h.createNew,args:Object.assign(Object.assign({},t),{isPalette:!0}),category:o})},e.addMenuItems=function(e,t,n,o,a,r){$(e,n),U(e,n),K(e,t,n,o),a&&J(e,t,n,a,o,r)},e.addKernelLanguageMenuItems=function(e,t){for(let n of t)e.fileMenu.newMenu.addItem({command:h.createNew,args:n,rank:31})},e.addUndoRedoToEditMenu=$,e.addEditorViewerToViewMenu=U,e.addConsoleCreatorToFileMenu=K,e.addCodeRunnersToRunMenu=J}(S||(S={}));const T={activate:function(e,t,n,o,r,d,i,l,s,u,m,g){const f=T.id,p=r.load("jupyterlab");let C;g&&(C=(0,a.createToolbarFactory)(g,o,_,f,r));const b=new c.FileEditorFactory({editorServices:t,factoryOptions:{name:_,fileTypes:["markdown","*"],defaultFor:["markdown","*"],toolbarFactory:C,translator:r}}),{commands:v,restored:h,shell:w}=e,k=new a.WidgetTracker({namespace:"editor"}),x=new Map([["python",[{fileExt:"py",iconName:"ui-components:python",launcherLabel:p.__("Python File"),paletteLabel:p.__("New Python File"),caption:p.__("Create a new Python file")}]],["julia",[{fileExt:"jl",iconName:"ui-components:julia",launcherLabel:p.__("Julia File"),paletteLabel:p.__("New Julia File"),caption:p.__("Create a new Julia file")}]],["R",[{fileExt:"r",iconName:"ui-components:r-kernel",launcherLabel:p.__("R File"),paletteLabel:p.__("New R File"),caption:p.__("Create a new R file")}]]]);return u&&u.restore(k,{command:"docmanager:open",args:e=>({path:e.context.path,factory:_}),name:e=>e.context.path}),Promise.all([o.load(f),h]).then((([e])=>{S.updateSettings(e,v),S.updateTracker(k),e.changed.connect((()=>{S.updateSettings(e,v),S.updateTracker(k)}))})).catch((e=>{console.error(e.message),S.updateTracker(k)})),b.widgetCreated.connect(((e,t)=>{t.context.pathChanged.connect((()=>{k.save(t)})),k.add(t),S.updateWidget(t.content)})),e.docRegistry.addWidgetFactory(b),k.widgetAdded.connect(((e,t)=>{S.updateWidget(t.content)})),S.addCommands(v,o,p,f,(()=>null!==k.currentWidget&&k.currentWidget===w.currentWidget),k,n),l&&S.addLauncherItems(l,p),i&&S.addPaletteItems(i,p),s&&S.addMenuItems(s,v,k,p,d,m),(async()=>{var t,n;const o=e.serviceManager.kernelspecs;await o.ready;let a=new Set;const r=null!==(n=null===(t=o.specs)||void 0===t?void 0:t.kernelspecs)&&void 0!==n?n:{};return Object.keys(r).forEach((e=>{const t=r[e];if(t){const e=x.get(t.language);null==e||e.forEach((e=>a.add(e)))}})),a})().then((e=>{l&&S.addKernelLanguageLauncherItems(l,p,e),i&&S.addKernelLanguagePaletteItems(i,p,e),s&&S.addKernelLanguageMenuItems(s,e)})).catch((e=>{console.error(e.message)})),k},id:"@jupyterlab/fileeditor-extension:plugin",requires:[r.IEditorServices,i.IFileBrowserFactory,u.ISettingRegistry,g.ITranslator],optional:[d.IConsoleTracker,a.ICommandPalette,l.ILauncher,s.IMainMenu,o.ILayoutRestorer,a.ISessionContextDialogs,a.IToolbarWidgetRegistry],provides:c.IEditorTracker,autoStart:!0},y={id:"@jupyterlab/fileeditor-extension:tab-space-status",autoStart:!0,requires:[c.IEditorTracker,u.ISettingRegistry,g.ITranslator],optional:[m.IStatusBar],activate:(e,t,n,o,a)=>{const d=o.load("jupyterlab");if(!a)return;const i=new f.Menu({commands:e.commands}),l="fileeditor:change-tabs",{shell:s}=e,u={insertSpaces:!1,size:4,name:d.__("Indent with Tab")};i.addItem({command:l,args:u});for(const e of[1,2,4,8]){const t={insertSpaces:!0,size:e,name:d._n("Spaces: %1","Spaces: %1",e)};i.addItem({command:l,args:t})}const m=new c.TabSpaceStatus({menu:i,translator:o}),g=e=>{m.model.config=Object.assign(Object.assign({},r.CodeEditor.defaultConfig),e.get("editorConfig").composite)};Promise.all([n.load("@jupyterlab/fileeditor-extension:plugin"),e.restored]).then((([e])=>{g(e),e.changed.connect(g)})),a.registerStatusItem("@jupyterlab/fileeditor-extension:tab-space-status",{item:m,align:"right",rank:1,isActive:()=>!!s.currentWidget&&t.has(s.currentWidget)})}},I=[T,y]}}]);
//# sourceMappingURL=4805.7300464.js.map