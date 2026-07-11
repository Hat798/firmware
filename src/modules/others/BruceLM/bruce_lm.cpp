/**
 * @file bruce_lm.cpp
 * @brief "Others -> BruceLM" module: on-device LLM.
 *
 *  ---- Made by @Doominator1 on GitHub, Jul 2026
 *  ---- Much thanks to @karpathy for the base of this project; llama2.c!
 *
 * Models + tokenizers live on SD at /BruceLM/models/ (auto-created here,
 * matching the same idiom used by ir_read.cpp's /BruceIR and
 * wardriving.cpp's /BruceWardriving). Model selection reuses Bruce's
 * existing loopSD() file picker; text entry reuses the existing keyboard()
 * overlay. Everything is drawn with the shared theme colors/border helpers
 * so it matches the rest of the firmware.
 *
 * Layout (top to bottom, inside the standard Bruce border+title):
 *   - scrolling chat transcript, word-wrapped, "You:" lines in the primary
 *     theme color and assistant text in the secondary color
 *   - a horizontal separator line
 *   - a one-line "input bar" row showing the current/last prompt
 *   - a footer hint row ("OK: new prompt   Esc: exit")
 *
 * The whole frame is redrawn from scratch on every change (new token,
 * scroll tick, new turn) instead of incrementally patching the screen -
 * slightly more pixels pushed, but it removes an entire class of stale-text/
 * overlap bugs that incremental cursor-based drawing is prone to.
 *
 * Scrolling uses this board's rotary encoder, which the board's InputHandler
 * (boards/lilygo-t-embed-cc1101/interface.cpp) maps to NextPress/PrevPress -
 * a separate signal from the physical Select/Back buttons (SelPress/EscPress),
 * so scrolling and cancelling never conflict. NextPress scrolls down toward
 * newer content, PrevPress scrolls up toward older content, matching how
 * Next/Prev already mean forward/backward everywhere else in this firmware.
 */
#include "bruce_lm.h"

#include "core/display.h"
#include "core/mykeyboard.h"
#include "core/sd_functions.h"
#include "llm_engine.h"
#include <ArduinoJson.h>
#include <globals.h>
#include <vector>

using namespace bruce_llm;

namespace {

constexpr const char *kRootDir = "/BruceLM";
constexpr const char *kModelsDir = "/BruceLM/models";
constexpr const char *kConfigPath = "/BruceLM/config.json";

// Seed presets cycled through with Next/Prev on the Seed row, rather than
// free-form numeric entry - "Random" reseeds from the hardware RNG each
// reply (value 0 is the sentinel llm_engine.cpp treats as "reseed"); the
// rest give byte-for-byte reproducible output for the same prompt.
struct SeedPreset {
    const char *label;
    uint32_t value;
};
constexpr SeedPreset kSeedPresets[] = {
    {"1",      1   },
    {"2",      2   },
    {"3",      3   },
    {"4",      4   },
    {"5",      5   },
    {"6",      6   },
    {"7",      7   },
    {"8",      8   },
    {"9",      9   },
    {"10",     10  },
    {"42",     42  },
    {"67",     67  },
    {"100",    100 },
    {"1000",   1000},
    {"Random", 0   },
};
constexpr int kNumSeedPresets = sizeof(kSeedPresets) / sizeof(kSeedPresets[0]);
constexpr int kDefaultSeedPresetIndex = kNumSeedPresets - 1; // "Random"

// Every field here is a real, functioning knob on the engine
// (llm_engine.cpp's generate() / GenerationParams) - nothing decorative.
// temperature/topP/seed are the same sampling controls run.c itself exposes;
// repetitionPenalty is a common addition upstream llama2.c doesn't have,
// included because it measurably helps tiny models avoid repetition loops.
struct BruceLMSettings {
    float temperature = 0.8f;
    float topP = 0.9f;
    float repetitionPenalty = 1.0f;
    int maxTokens = 256;
    int seedPresetIndex = kDefaultSeedPresetIndex;

    uint32_t seed() const { return kSeedPresets[seedPresetIndex].value; }
};

BruceLMSettings loadSettings(FS &fs) {
    BruceLMSettings s;
    if (!fs.exists(kConfigPath)) return s;
    File f = fs.open(kConfigPath, FILE_READ);
    if (!f) return s;
    JsonDocument doc;
    if (!deserializeJson(doc, f)) {
        if (!doc["temperature"].isNull()) s.temperature = doc["temperature"];
        if (!doc["topP"].isNull()) s.topP = doc["topP"];
        if (!doc["repetitionPenalty"].isNull()) s.repetitionPenalty = doc["repetitionPenalty"];
        if (!doc["maxTokens"].isNull()) s.maxTokens = doc["maxTokens"];
        if (!doc["seedPresetIndex"].isNull()) {
            int idx = doc["seedPresetIndex"];
            if (idx >= 0 && idx < kNumSeedPresets) s.seedPresetIndex = idx;
        }
    }
    f.close();
    return s;
}

void saveSettings(FS &fs, const BruceLMSettings &s) {
    JsonDocument doc;
    doc["temperature"] = s.temperature;
    doc["topP"] = s.topP;
    doc["repetitionPenalty"] = s.repetitionPenalty;
    doc["maxTokens"] = s.maxTokens;
    doc["seedPresetIndex"] = s.seedPresetIndex;
    File f = fs.open(kConfigPath, FILE_WRITE);
    if (!f) return;
    serializeJsonPretty(doc, f);
    f.close();
}

// All of this is deliberately expressed in terms of existing global layout
// constants (BORDER_PAD_X/Y, FP/FM, LW/LH) rather than magic numbers, and the
// outer rounded-rect border is drawn at (5,5,tftWidth-10,tftHeight-10) by
// drawStatusBar() - kBottomMargin/kTopGap keep our own content clear of it.
constexpr int kTopGap = 6;
constexpr int kBottomMargin = 6;
constexpr int kFooterH = 16;
constexpr int kInputBarH = 16;
constexpr int kSepGap = 4;

struct ChatGeometry {
    int outputTop;
    int sepY;
    int inputBarY;
    int footerY;
    int lineH;
    int maxChars;
    int visibleLines;
};

ChatGeometry computeGeometry() {
    ChatGeometry g;
    g.outputTop = BORDER_PAD_Y + FM * LH + kTopGap;
    g.footerY = tftHeight - kBottomMargin - kFooterH;
    g.inputBarY = g.footerY - kInputBarH;
    g.sepY = g.inputBarY - kSepGap;
    g.lineH = FP * LH;
    g.maxChars = (tftWidth - 2 * BORDER_PAD_X) / (FP * LW);
    if (g.maxChars < 1) g.maxChars = 1;
    g.visibleLines = (g.sepY - g.outputTop) / g.lineH;
    if (g.visibleLines < 1) g.visibleLines = 1;
    return g;
}

// Greedy word-wrap of a single line (no embedded '\n'). Always emits at
// least one (possibly empty) line, so blank paragraphs still take up a row.
std::vector<String> wrapLine(const String &text, int maxChars) {
    std::vector<String> out;
    int n = text.length();
    int start = 0;
    String currentLine;
    while (start < n) {
        int spaceIdx = text.indexOf(' ', start);
        int wordEnd = (spaceIdx == -1) ? n : spaceIdx;
        String word = text.substring(start, wordEnd);
        while ((int)word.length() > maxChars) {
            if (currentLine.length() > 0) {
                out.push_back(currentLine);
                currentLine = "";
            }
            out.push_back(word.substring(0, maxChars));
            word = word.substring(maxChars);
        }
        if (currentLine.length() == 0) currentLine = word;
        else if ((int)(currentLine.length() + 1 + word.length()) <= maxChars) currentLine += " " + word;
        else {
            out.push_back(currentLine);
            currentLine = word;
        }
        start = (spaceIdx == -1) ? n : spaceIdx + 1;
    }
    out.push_back(currentLine);
    return out;
}

// Splits on explicit '\n' first (the model can emit these), then word-wraps
// each resulting segment independently.
std::vector<String> wrapText(const String &text, int maxChars) {
    std::vector<String> out;
    int start = 0;
    int n = text.length();
    for (;;) {
        int nl = text.indexOf('\n', start);
        String segment = (nl == -1) ? text.substring(start) : text.substring(start, nl);
        for (auto &l : wrapLine(segment, maxChars)) out.push_back(l);
        if (nl == -1) break;
        start = nl + 1;
    }
    return out;
}

struct ChatLine {
    String text;
    bool isUser;
};

struct ChatSession {
    // Each entry is one message ("You: ..." or the assistant's reply text,
    // appended to as tokens stream in). true = user message.
    std::vector<std::pair<bool, String>> messages;
    int scrollOffset = 0;
    bool followBottom = true;
    String inputBarPreview;
    BruceLMSettings settings;
};

std::vector<ChatLine> buildLines(const ChatSession &s, int maxChars) {
    std::vector<ChatLine> out;
    for (size_t i = 0; i < s.messages.size(); i++) {
        bool isUser = s.messages[i].first;
        for (auto &l : wrapText(s.messages[i].second, maxChars)) out.push_back({l, isUser});
        if (i + 1 < s.messages.size()) out.push_back({"", false}); // blank line between messages
    }
    return out;
}

// Applies rotary-encoder scroll input. Returns true if the visible content
// actually needs to move as a result.
bool handleScrollInput(ChatSession &s, int totalLines, int visibleLines) {
    int maxScroll = totalLines > visibleLines ? totalLines - visibleLines : 0;
    bool changed = false;
    if (check(NextPress)) { // rotate toward newer/bottom content
        if (s.scrollOffset < maxScroll) {
            s.scrollOffset++;
            changed = true;
        }
        s.followBottom = (s.scrollOffset >= maxScroll);
    }
    if (check(PrevPress)) { // rotate toward older/top content
        if (s.scrollOffset > 0) {
            s.scrollOffset--;
            changed = true;
        }
        s.followBottom = false;
    }
    return changed;
}

void clampScroll(ChatSession &s, int totalLines, int visibleLines) {
    int maxScroll = totalLines > visibleLines ? totalLines - visibleLines : 0;
    if (s.followBottom) s.scrollOffset = maxScroll;
    if (s.scrollOffset > maxScroll) s.scrollOffset = maxScroll;
    if (s.scrollOffset < 0) s.scrollOffset = 0;
}

String truncateToWidth(const String &text, int maxChars) {
    if ((int)text.length() <= maxChars) return text;
    if (maxChars <= 3) return text.substring(0, maxChars);
    return text.substring(0, maxChars - 3) + "...";
}

void drawChatFrame(const ChatGeometry &g, const String &inputBarPreview) {
    drawMainBorderWithTitle("BruceLM");

    tft.drawFastHLine(BORDER_PAD_X, g.sepY, tftWidth - 2 * BORDER_PAD_X, bruceConfig.priColor);

    tft.setTextColor(bruceConfig.secColor, bruceConfig.bgColor);
    int inputTextY = g.inputBarY + (kInputBarH - g.lineH) / 2;
    tft.setCursor(BORDER_PAD_X, inputTextY);
    String preview = inputBarPreview.length() ? ("> " + inputBarPreview) : "> Tap OK to type a prompt";
    tft.print(truncateToWidth(preview, g.maxChars));

    tft.setTextColor(bruceConfig.priColor, bruceConfig.bgColor);
    String hint = "OK: new prompt  Esc: exit";
    int hintTextY = g.footerY + (kFooterH - g.lineH) / 2;
    int hx = (tftWidth - (int)(hint.length() * FP * LW)) / 2;
    if (hx < BORDER_PAD_X) hx = BORDER_PAD_X;
    tft.setCursor(hx, hintTextY);
    tft.print(hint);
}

void drawVisibleLines(const std::vector<ChatLine> &lines, int scrollOffset, const ChatGeometry &g) {
    for (int i = 0; i < g.visibleLines; i++) {
        int idx = scrollOffset + i;
        if (idx >= (int)lines.size()) break;
        tft.setTextColor(
            lines[idx].isUser ? bruceConfig.priColor : bruceConfig.secColor, bruceConfig.bgColor
        );
        tft.setCursor(BORDER_PAD_X, g.outputTop + i * g.lineH);
        tft.print(lines[idx].text);
    }
}

// Single entry point for both drawing and encoder-scroll handling, so the
// idle loop and the token-streaming loop can never disagree about how
// scrolling behaves. Returns true if it actually redrew the screen.
bool renderChat(ChatSession &s, bool force) {
    ChatGeometry g = computeGeometry();
    std::vector<ChatLine> lines = buildLines(s, g.maxChars);
    int total = (int)lines.size();

    bool scrolled = handleScrollInput(s, total, g.visibleLines);
    clampScroll(s, total, g.visibleLines);

    if (!force && !scrolled) return false;

    drawChatFrame(g, s.inputBarPreview);
    drawVisibleLines(lines, s.scrollOffset, g);
    return true;
}

void ensureFolders(FS &fs) {
    if (!folderExists(fs, kRootDir)) createFolder(fs, kRootDir);
    if (!folderExists(fs, kModelsDir)) createFolder(fs, kModelsDir);
}

// Draws a message starting at the top of the content area, an optional
// standout note centered in the gap between that text and the footer (for
// calling out the required action - e.g. "Press OK to..." - so it isn't
// just a small line buried in the bottom hint), and a hint row pinned to
// the bottom (same geometry the chat frame uses) so button mappings always
// land in the same place instead of wherever the message text happened to end.
void drawMessageScreen(const String &msg, const String &hint, const String &centerNote = "") {
    ChatGeometry g = computeGeometry();
    drawMainBorderWithTitle("BruceLM");

    tft.setTextColor(bruceConfig.priColor, bruceConfig.bgColor);
    int y = g.outputTop;
    for (auto &line : wrapText(msg, g.maxChars)) {
        tft.setCursor(BORDER_PAD_X, y);
        tft.print(line);
        y += g.lineH;
    }

    if (centerNote.length() > 0) {
        int aboveFooterY = g.footerY - 4;
        int noteY = y + max(0, (aboveFooterY - y - g.lineH) / 2);
        tft.setTextColor(TFT_YELLOW, bruceConfig.bgColor);
        for (auto &line : wrapText(centerNote, g.maxChars)) {
            int nx = (tftWidth - (int)(line.length() * FP * LW)) / 2;
            if (nx < BORDER_PAD_X) nx = BORDER_PAD_X;
            tft.setCursor(nx, noteY);
            tft.print(line);
            noteY += g.lineH;
        }
    }

    tft.setTextColor(bruceConfig.priColor, bruceConfig.bgColor);
    int hintTextY = g.footerY + (kFooterH - g.lineH) / 2;
    int hx = (tftWidth - (int)(hint.length() * FP * LW)) / 2;
    if (hx < BORDER_PAD_X) hx = BORDER_PAD_X;
    tft.setCursor(hx, hintTextY);
    tft.print(hint);
}

void showMessageAndWait(const String &msg) {
    drawMessageScreen(msg, "Esc: return");
    while (!check(EscPress)) delay(30);
}

// Blocks until the user picks Select (true) or Esc (false).
bool showConfirm(const String &msg, const String &hint, const String &centerNote = "") {
    drawMessageScreen(msg, hint, centerNote);
    for (;;) {
        if (check(SelPress)) return true;
        if (check(EscPress)) return false;
        delay(30);
    }
}

float adjustFloat(float v, float step, float lo, float hi, int dir) {
    v += step * dir;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

int adjustInt(int v, int step, int lo, int hi, int dir) {
    v += step * dir;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

int cycleIndex(int idx, int count, int dir) { return (idx + dir + count) % count; }

struct SettingsEditState {
    bool editing = false;
    int editRow = -1;
    float tempBackup = 0, topPBackup = 0, repPenBackup = 0;
    int maxTokBackup = 0, seedIdxBackup = 0;
};

void renderSettingsRows(
    const BruceLMSettings &s, int cursor, const SettingsEditState &edit, int startY, int rowH
) {
    struct Row {
        const char *label;
        String value;
    } rows[5] = {
        {"Temperature", String(s.temperature, 1)},
        {"Top-p", String(s.topP, 2)},
        {"Rep. Penalty", String(s.repetitionPenalty, 2)},
        {"Max Tokens", String(s.maxTokens)},
        {"Seed", String(kSeedPresets[s.seedPresetIndex].label)},
    };

    tft.setTextSize(FP);
    for (int i = 0; i < 5; i++) {
        int rowY = startY + i * rowH;
        bool selected = (cursor == i);
        bool editingThis = (edit.editing && edit.editRow == i);

        tft.fillRect(BORDER_PAD_X, rowY, tftWidth - 2 * BORDER_PAD_X, rowH, bruceConfig.bgColor);
        tft.setTextColor(selected ? TFT_YELLOW : bruceConfig.priColor, bruceConfig.bgColor);
        tft.setCursor(BORDER_PAD_X, rowY + 2);
        tft.print(rows[i].label);

        String valueText = editingThis ? ("[ " + rows[i].value + " ]") : rows[i].value;
        int vx = tftWidth - BORDER_PAD_X - (int)(valueText.length() * FP * LW);
        tft.setCursor(vx, rowY + 2);
        tft.print(valueText);
    }
}

// ble_spam.cpp-style row editor: Next/Prev moves the cursor between rows,
// Select enters/exits edit mode on the highlighted row, Next/Prev while
// editing adjusts that row's value, Esc while editing reverts to the value
// it had before editing started, Esc while browsing saves and exits.
void showSettingsScreen(FS &fs) {
    BruceLMSettings s = loadSettings(fs);
    SettingsEditState edit;
    int cursor = 0;
    ChatGeometry g = computeGeometry();
    int rowH = g.lineH + 4;
    int startY = g.outputTop;

    bool redraw = true;
    drawMainBorderWithTitle("BruceLM Settings");

    constexpr int kSaveRow = 5;  // bottom action row, ble_spam "[ Start ]"-style
    constexpr int kNumItems = 6; // 5 editable rows + save

    for (;;) {
        if (redraw) {
            renderSettingsRows(s, cursor, edit, startY, rowH);

            int saveY = startY + kSaveRow * rowH;
            tft.fillRect(BORDER_PAD_X, saveY, tftWidth - 2 * BORDER_PAD_X, rowH, bruceConfig.bgColor);
            tft.setTextColor(cursor == kSaveRow ? TFT_YELLOW : bruceConfig.priColor, bruceConfig.bgColor);
            String saveLabel = "[ Save ]";
            int sx = (tftWidth - (int)(saveLabel.length() * FP * LW)) / 2;
            tft.setCursor(sx, saveY + 2);
            tft.print(saveLabel);

            tft.setTextColor(bruceConfig.priColor, bruceConfig.bgColor);
            String hint = edit.editing ? "OK: confirm  Esc: revert" : "OK: edit  Esc: save+exit";
            int hintTextY = g.footerY + (kFooterH - g.lineH) / 2;
            tft.fillRect(0, hintTextY - 2, tftWidth, g.lineH + 4, bruceConfig.bgColor);
            int hx = (tftWidth - (int)(hint.length() * FP * LW)) / 2;
            if (hx < BORDER_PAD_X) hx = BORDER_PAD_X;
            tft.setCursor(hx, hintTextY);
            tft.print(hint);
            redraw = false;
        }

        if (check(EscPress)) {
            // Esc always means "save and exit" from the row list (or revert
            // while editing) - unchanged regardless of the Save row below.
            if (edit.editing) {
                switch (edit.editRow) {
                    case 0: s.temperature = edit.tempBackup; break;
                    case 1: s.topP = edit.topPBackup; break;
                    case 2: s.repetitionPenalty = edit.repPenBackup; break;
                    case 3: s.maxTokens = edit.maxTokBackup; break;
                    case 4: s.seedPresetIndex = edit.seedIdxBackup; break;
                }
                edit.editing = false;
                redraw = true;
            } else {
                saveSettings(fs, s);
                return;
            }
        } else if (check(SelPress)) {
            if (edit.editing) {
                edit.editing = false;
            } else if (cursor == kSaveRow) {
                saveSettings(fs, s);
                return;
            } else {
                edit.editing = true;
                edit.editRow = cursor;
                edit.tempBackup = s.temperature;
                edit.topPBackup = s.topP;
                edit.repPenBackup = s.repetitionPenalty;
                edit.maxTokBackup = s.maxTokens;
                edit.seedIdxBackup = s.seedPresetIndex;
            }
            redraw = true;
        } else if (edit.editing) {
            int dir = check(NextPress) ? 1 : (check(PrevPress) ? -1 : 0);
            if (dir != 0) {
                switch (edit.editRow) {
                    case 0: s.temperature = adjustFloat(s.temperature, 0.1f, 0.0f, 2.0f, dir); break;
                    case 1: s.topP = adjustFloat(s.topP, 0.05f, 0.0f, 1.0f, dir); break;
                    case 2:
                        s.repetitionPenalty = adjustFloat(s.repetitionPenalty, 0.05f, 1.0f, 2.0f, dir);
                        break;
                    case 3: s.maxTokens = adjustInt(s.maxTokens, 16, 16, 2048, dir); break;
                    case 4: s.seedPresetIndex = cycleIndex(s.seedPresetIndex, kNumSeedPresets, dir); break;
                }
                redraw = true;
            }
        } else if (check(NextPress)) {
            cursor = (cursor + 1) % kNumItems;
            redraw = true;
        } else if (check(PrevPress)) {
            cursor = (cursor + kNumItems - 1) % kNumItems;
            redraw = true;
        }

        delay(30);
    }
}

enum class StartAction { Start, Settings, Exit };

// First screen: short concrete instructions above two selectable action
// rows ("Start" / "Settings"), same cursor+Next/Prev+Select style as the
// settings screen above, with the button hints pinned to the bottom.
StartAction showStartScreen() {
    int cursor = 0;
    const char *labels[2] = {"Start", "Settings"};
    ChatGeometry g = computeGeometry();
    String instructions = "1. Get a model + tokenizer -\n"
                          "e.g. stories260K.bin + tok512.bin\n"
                          "from huggingface.co/karpathy/\n"
                          "tinyllamas\n"
                          "2. Copy both to BruceLM/models";
    std::vector<String> instructionLines = wrapText(instructions, g.maxChars);

    int rowH = g.lineH + 6;
    // Centered in the gap between the end of the instructions text and the
    // footer hint row, rather than right under the text - clamped so it can
    // never creep up into the text if the instructions are ever lengthened.
    int afterTextY = g.outputTop + (int)instructionLines.size() * g.lineH + 4;
    int aboveFooterY = g.footerY - 4;
    int rowsBlockH = rowH * 2;
    int row1Y = afterTextY + max(0, (aboveFooterY - afterTextY - rowsBlockH) / 2);
    int row2Y = row1Y + rowH;

    bool redraw = true;
    for (;;) {
        if (redraw) {
            drawMainBorderWithTitle("BruceLM");
            tft.setTextColor(bruceConfig.priColor, bruceConfig.bgColor);
            int y = g.outputTop;
            for (auto &line : instructionLines) {
                tft.setCursor(BORDER_PAD_X, y);
                tft.print(line);
                y += g.lineH;
            }

            for (int i = 0; i < 2; i++) {
                int rowY = (i == 0) ? row1Y : row2Y;
                tft.fillRect(BORDER_PAD_X, rowY, tftWidth - 2 * BORDER_PAD_X, rowH, bruceConfig.bgColor);
                tft.setTextColor(cursor == i ? TFT_YELLOW : bruceConfig.priColor, bruceConfig.bgColor);
                String label = String("> ") + labels[i];
                int lx = (tftWidth - (int)(label.length() * FP * LW)) / 2;
                tft.setCursor(lx, rowY + 2);
                tft.print(label);
            }

            tft.setTextColor(bruceConfig.priColor, bruceConfig.bgColor);
            String hint = "OK: select  Esc: exit";
            int hintTextY = g.footerY + (kFooterH - g.lineH) / 2;
            int hx = (tftWidth - (int)(hint.length() * FP * LW)) / 2;
            if (hx < BORDER_PAD_X) hx = BORDER_PAD_X;
            tft.setCursor(hx, hintTextY);
            tft.print(hint);
            redraw = false;
        }

        if (check(EscPress)) return StartAction::Exit;
        if (check(SelPress)) return cursor == 0 ? StartAction::Start : StartAction::Settings;
        if (check(NextPress) || check(PrevPress)) {
            cursor = 1 - cursor;
            redraw = true;
        }
        delay(30);
    }
}

void showLoadingScreen(const String &modelFileName, const String &tokenizerFileName) {
    ChatGeometry g = computeGeometry();
    drawMainBorderWithTitle("BruceLM");
    tft.setTextColor(bruceConfig.priColor, bruceConfig.bgColor);
    tft.setCursor(BORDER_PAD_X, g.outputTop);
    tft.print("Loading model + tokenizer...");
    tft.setTextColor(bruceConfig.secColor, bruceConfig.bgColor);
    tft.setCursor(BORDER_PAD_X, g.outputTop + g.lineH * 2);
    tft.print(truncateToWidth(modelFileName, g.maxChars));
    tft.setCursor(BORDER_PAD_X, g.outputTop + g.lineH * 3);
    tft.print(truncateToWidth(tokenizerFileName, g.maxChars));
}

// Streams the transcript into the top output area; returns false if the user
// cancelled generation (Esc) partway through.
bool runChatTurn(ChatSession &session, LLMEngine &engine, const String &userPrompt) {
    session.messages.push_back({true, "You: " + userPrompt});
    session.messages.push_back({false, ""});
    session.inputBarPreview = userPrompt;
    session.followBottom = true;
    renderChat(session, /*force=*/true);

    GenerationParams params;
    params.temperature = session.settings.temperature;
    params.topP = session.settings.topP;
    params.repetitionPenalty = session.settings.repetitionPenalty;
    params.seed = session.settings.seed();

    bool cancelled = false;
    engine.generate(userPrompt, session.settings.maxTokens, params, [&](const String &piece) -> bool {
        // Generation can run longer than Bruce's normal screen-timeout
        // window. wakeUpScreen() just resets its inactivity clock (the
        // same thing any real button press does) - calling it here from
        // our own module, not touching core/display.cpp itself, keeps
        // the screen from dimming/sleeping mid-reply.
        wakeUpScreen();
        if (check(EscPress)) {
            cancelled = true;
            return false;
        }
        session.messages.back().second += piece;
        renderChat(session, /*force=*/true);
        return true;
    });

    if (cancelled) {
        session.messages.back().second += " [cancelled]";
        renderChat(session, /*force=*/true);
    }
    return !cancelled;
}

void chatLoop(LLMEngine &engine, FS &fs, const BruceLMSettings &settings) {
    ChatSession session;
    session.settings = settings;
    renderChat(session, /*force=*/true);

    for (;;) {
        if (check(EscPress)) return;
        if (check(SelPress)) {
            String prompt = keyboard("", 200, "Prompt:");
            String trimmed = prompt;
            trimmed.trim();
            trimmed.toLowerCase();
            if (trimmed == "/settings") {
                // No spare physical input to dedicate to "open settings" while
                // chatting (Next/Prev already scroll, Select starts a prompt,
                // Esc exits) - reusing the existing prompt entry as a command
                // avoids adding a new input mapping.
                showSettingsScreen(fs);
                session.settings = loadSettings(fs);
                renderChat(session, /*force=*/true);
            } else if (prompt.length() > 0) {
                runChatTurn(session, engine, prompt);
            }
        } else {
            renderChat(session, /*force=*/false); // only redraws if the encoder scrolled
        }
        delay(30);
    }
}

// IncompatibleGroupSize and ConfigTooLarge are both "this will probably go
// badly, but try anyway?" situations - one confirm+retry path for both
// instead of duplicating the same dialog/retry logic per error code.
LLMLoadError confirmRiskyLoad(
    FS &fs, const String &checkpointPath, const String &tokenizerPath, LLMEngine &engine, LLMLoadError err,
    const String &modelFileName, const String &tokenizerFileName
) {
    String warning;
    if (err == LLMLoadError::IncompatibleGroupSize) {
        warning = "This model's quantization group\n"
                  "size doesn't evenly divide its\n"
                  "layer sizes - output will likely\n"
                  "come out garbled.";
    } else if (err == LLMLoadError::ConfigTooLarge) {
        warning = "This model looks too large for\n"
                  "available memory - it may fail to\n"
                  "load or crash mid-reply.";
    } else {
        return err;
    }

    if (!showConfirm(warning, "OK: run anyway   Esc: cancel")) return err;
    showLoadingScreen(modelFileName, tokenizerFileName);
    return engine.load(fs, checkpointPath, tokenizerPath, /*overrideSafetyChecks=*/true);
}

} // namespace

void bruceLM_setup() {
    FS *fs;
    if (!getFsStorage(fs)) {
        showMessageAndWait("No storage available (SD/LittleFS).");
        return;
    }

    ensureFolders(*fs);

    for (;;) {
        StartAction action = showStartScreen();
        if (action == StartAction::Exit) return;
        if (action == StartAction::Settings) {
            showSettingsScreen(*fs);
            continue;
        }
        break; // StartAction::Start
    }

    BruceLMSettings settings = loadSettings(*fs);

    if (!showConfirm(
            "Select a model checkpoint file.\n"
            "\n"
            "Recommended example:\n"
            "\n"
            "stories260K.bin",
            "OK: continue   Esc: cancel",
            "Press OK to open the file picker"
        ))
        return;
    String checkpointPath = loopSD(*fs, true, "bin", kModelsDir);
    if (checkpointPath.length() == 0) return; // user backed out of the picker

    if (!showConfirm(
            "Now select the tokenizer file\n"
            "that matches your model.\n"
            "\n"
            "Recommended example:\n"
            "\n"
            "tok512.bin",
            "OK: continue   Esc: cancel",
            "Press OK to open the file picker"
        ))
        return;
    String tokenizerPath = loopSD(*fs, true, "bin", kModelsDir);
    if (tokenizerPath.length() == 0) return; // user backed out of the picker

    String modelFileName = checkpointPath.substring(checkpointPath.lastIndexOf('/') + 1);
    String tokenizerFileName = tokenizerPath.substring(tokenizerPath.lastIndexOf('/') + 1);
    showLoadingScreen(modelFileName, tokenizerFileName);

    LLMEngine engine;
    LLMLoadError err = engine.load(*fs, checkpointPath, tokenizerPath);

    if (err == LLMLoadError::IncompatibleGroupSize || err == LLMLoadError::ConfigTooLarge)
        err = confirmRiskyLoad(
            *fs, checkpointPath, tokenizerPath, engine, err, modelFileName, tokenizerFileName
        );

    switch (err) {
        case LLMLoadError::None: break;
        case LLMLoadError::CheckpointNotFound:
        case LLMLoadError::TokenizerNotFound:
            showMessageAndWait("Model or tokenizer file went missing.");
            return;
        case LLMLoadError::BadMagicOrVersion:
            showMessageAndWait(
                "Unrecognized checkpoint format.\n"
                "Expected a llama2.c export.py v1/v2\n"
                "file, or the original header-less\n"
                "format karpathy/tinyllamas ships."
            );
            return;
        case LLMLoadError::ConfigTooLarge:
            showMessageAndWait("Could not load: model too large for available memory.");
            return;
        case LLMLoadError::OutOfMemory:
            showMessageAndWait("Out of memory while allocating model buffers.");
            return;
        case LLMLoadError::IncompatibleGroupSize:
            showMessageAndWait("Could not load: incompatible quantized model.");
            return;
    }

    chatLoop(engine, *fs, settings);
}
