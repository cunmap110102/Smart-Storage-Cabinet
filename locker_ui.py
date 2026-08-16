import os
import csv
import json
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import ttk
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from camera_utils import create_camera
from capture_data import (
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    FRAME_PROCESS_SCALE,
    FaceCaptureSession,
    classify_face_state,
    detect_face_boxes,
    is_face_frontal_enough,
)
from fingerprint_controller import FingerprintController
from relay_controller import RelayController
from recognize_face import FaceRecognitionSession
from svm_model import load_state_svm
from train_model import (
    delete_user,
    extract_original_face_vectors,
    inspect_model_consistency,
    inspect_identity_separation,
    load_model,
    train_user,
)


def _get_bool_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _adjust_hex_color(hex_color, factor):
    hex_value = str(hex_color or "#000000").lstrip("#")
    if len(hex_value) != 6:
        return hex_color
    red = int(hex_value[0:2], 16)
    green = int(hex_value[2:4], 16)
    blue = int(hex_value[4:6], 16)
    if factor >= 0:
        red = int(red + (255 - red) * factor)
        green = int(green + (255 - green) * factor)
        blue = int(blue + (255 - blue) * factor)
    else:
        red = int(red * (1 + factor))
        green = int(green * (1 + factor))
        blue = int(blue * (1 + factor))
    red = max(0, min(255, red))
    green = max(0, min(255, green))
    blue = max(0, min(255, blue))
    return f"#{red:02X}{green:02X}{blue:02X}"


def _hex_to_rgba(hex_color, alpha=255):
    hex_value = str(hex_color or "#000000").lstrip("#")
    if len(hex_value) != 6:
        return (0, 0, 0, alpha)
    return (
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
        alpha,
    )


class CanvasButton(tk.Canvas):
    def __init__(
        self,
        master,
        text,
        command,
        fill,
        fg="white",
        hover_fill=None,
        active_fill=None,
        disabled_fill="#9CA3AF",
        parent_bg="#F0F2F5",
        width=220,
        height=58,
        radius=18,
        font=None,
        subtext="",
        subfont=None,
        icon="",
        icon_font=None,
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=parent_bg,
            bd=0,
            highlightthickness=0,
            relief="flat",
            cursor="hand2",
        )
        self._text = text
        self._subtext = subtext
        self._icon = icon
        self._command = command
        self._fill = fill
        self._fg = fg
        self._hover_fill = hover_fill or _adjust_hex_color(fill, 0.08)
        self._active_fill = active_fill or _adjust_hex_color(fill, -0.12)
        self._disabled_fill = disabled_fill
        self._parent_bg = parent_bg
        self._radius = radius
        self._font = font
        self._subfont = subfont
        self._icon_font = icon_font or font
        self._enabled = True
        self._pressed = False
        self._hovered = False
        self._button_photo = None
        self._draw_after_id = None

        self.bind("<Configure>", lambda _event: self._schedule_draw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<space>", self._on_keyboard_activate)
        self.bind("<Return>", self._on_keyboard_activate)
        self.tag_bind("button", "<ButtonPress-1>", self._on_press)
        self.tag_bind("button", "<ButtonRelease-1>", self._on_release)
        self._draw()

    def configure(self, cnf=None, **kwargs):
        if cnf is None and not kwargs:
            return super().configure()
        if cnf:
            if isinstance(cnf, dict):
                kwargs.update(cnf)
            else:
                return super().configure(cnf, **kwargs)

        redraw = False
        state = kwargs.pop("state", None)
        if state is not None:
            self._enabled = str(state) != str(tk.DISABLED)
            redraw = True

        if "bg" in kwargs:
            self._fill = kwargs.pop("bg")
            self._hover_fill = _adjust_hex_color(self._fill, 0.08)
            self._active_fill = _adjust_hex_color(self._fill, -0.12)
            redraw = True
        if "background" in kwargs:
            self._fill = kwargs.pop("background")
            self._hover_fill = _adjust_hex_color(self._fill, 0.08)
            self._active_fill = _adjust_hex_color(self._fill, -0.12)
            redraw = True
        if "fg" in kwargs:
            self._fg = kwargs.pop("fg")
            redraw = True
        if "foreground" in kwargs:
            self._fg = kwargs.pop("foreground")
            redraw = True
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            redraw = True
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "cursor" in kwargs:
            super().configure(cursor=kwargs.pop("cursor"))
        if kwargs:
            super().configure(**kwargs)
            redraw = True
        if redraw:
            self._schedule_draw()

    config = configure

    def _current_fill(self):
        if not self._enabled:
            return self._disabled_fill
        if self._pressed:
            return self._active_fill
        if self._hovered:
            return self._hover_fill
        return self._fill

    def _schedule_draw(self):
        if self._draw_after_id is not None:
            return
        self._draw_after_id = self.after_idle(self._draw)

    def _draw(self):
        self._draw_after_id = None
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1:
            width = int(float(self.cget("width")))
        if height <= 1:
            height = int(float(self.cget("height")))
        width = max(1, int(width))
        height = max(1, int(height))
        fill = self._current_fill()
        body_width = max(1, width - 6)
        body_height = max(1, height - 7)
        radius = min(self._radius, body_height // 2, body_width // 2)

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (4, 5, width - 1, height - 1),
            radius=radius,
            fill=(15, 23, 42, 34 if self._enabled else 14),
        )

        gradient = Image.new("RGBA", (body_width, body_height), (0, 0, 0, 0))
        grad_draw = ImageDraw.Draw(gradient)
        top_color = _hex_to_rgba(_adjust_hex_color(fill, 0.12))
        bottom_color = _hex_to_rgba(_adjust_hex_color(fill, -0.04))
        for y in range(body_height):
            ratio = y / max(1, body_height - 1)
            red = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
            green = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
            blue = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
            grad_draw.line((0, y, body_width, y), fill=(red, green, blue, 255))

        mask = Image.new("L", (body_width, body_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle(
            (0, 0, body_width - 1, body_height - 1),
            radius=radius,
            fill=255,
        )
        image.paste(gradient, (0, 0), mask)
        draw.rounded_rectangle(
            (0, 0, body_width - 1, body_height - 1),
            radius=radius,
            outline=(255, 255, 255, 42),
            width=1,
        )

        self._button_photo = ImageTk.PhotoImage(image)
        self.delete("all")
        self.create_image(0, 0, image=self._button_photo, anchor="nw", tags=("button",))

        text_x = width // 2
        if self._icon:
            self.create_text(
                24,
                body_height // 2,
                text=self._icon,
                fill=self._fg,
                font=self._icon_font,
                anchor="center",
                tags=("button",),
            )
            text_x = max(58, width // 2 + 8)

        if self._subtext:
            self.create_text(
                text_x,
                body_height // 2 - 9,
                text=self._text,
                fill=self._fg,
                font=self._font,
                anchor="center",
                tags=("button",),
            )
            self.create_text(
                text_x,
                body_height // 2 + 14,
                text=self._subtext,
                fill=_adjust_hex_color(self._fg, -0.02) if self._fg.startswith("#") else self._fg,
                font=self._subfont,
                anchor="center",
                tags=("button",),
            )
        else:
            self.create_text(
                text_x,
                body_height // 2,
                text=self._text,
                fill=self._fg,
                font=self._font,
                anchor="center",
            tags=("button",),
        )
        self.create_rectangle(
            0,
            0,
            width,
            height,
            fill="",
            outline="",
            tags=("button", "hitbox"),
        )

    def _on_enter(self, _event):
        if not self._enabled:
            return
        self._hovered = True
        self._schedule_draw()

    def _on_leave(self, _event):
        self._hovered = False
        self._pressed = False
        self._schedule_draw()

    def _on_press(self, _event):
        if not self._enabled:
            return
        self._pressed = True
        self.focus_set()
        self._schedule_draw()

    def _on_release(self, event):
        if not self._enabled:
            return
        was_pressed = self._pressed
        self._pressed = False
        self._schedule_draw()
        if not was_pressed:
            return
        x_inside = 0 <= event.x <= self.winfo_width()
        y_inside = 0 <= event.y <= self.winfo_height()
        if x_inside and y_inside and self._command is not None:
            self._command()

    def _on_keyboard_activate(self, _event):
        if self._enabled and self._command is not None:
            self._command()
        return "break"


class LockerApp:
    LOCKER_IDS = ("01", "02", "03")
    AUTH_MODE_FACE_AND_FINGERPRINT = "face_and_fingerprint"
    AUTH_MODE_FACE_OR_FINGERPRINT = "face_or_fingerprint"
    AUTH_MODE_FACE_ONLY_PC = "face_only_pc"
    RECOGNITION_TIMEOUT_SECONDS = 3
    RELAY_PIN_MAP = {"01": 4, "02": 18, "03": 27}
    DOOR_SENSOR_PIN_MAP = {"01": 25, "02": 17, "03": 22}
    DOOR_SENSOR_CLOSED_ACTIVE_LOW = True
    STOP_RELAY_WHEN_DOOR_OPENS = True
    RELAY_MIN_ON_SECONDS_BEFORE_OPEN_CHECK = 0.8
    DOOR_OPEN_SIGNAL_SETTLE_SECONDS = 0.05
    RELAY_PULSE_POLL_SECONDS = 0.02
    DOOR_SENSOR_WAIT_CLOSE_SECONDS = 60
    DOOR_SCREEN_AUTO_RETURN_SECONDS = 10
    FINGERPRINT_STARTUP_RETRY_SECONDS = 30
    FINGERPRINT_STARTUP_RETRY_INTERVAL_MS = 2000
    TAKE_CONFIDENCE_THRESHOLD = 68.0
    TAKE_MASKED_CONFIDENCE_THRESHOLD = 70.0
    TAKE_REVIEW_CONFIDENCE_THRESHOLD = 58.0
    TAKE_REVIEW_MASKED_CONFIDENCE_THRESHOLD = 60.0
    TAKE_RECHECK_ACCEPT_CONFIDENCE_THRESHOLD = 65.0
    TAKE_RECHECK_ACCEPT_MASKED_CONFIDENCE_THRESHOLD = 61.0
    TAKE_REVIEW_TIMEOUT_SECONDS = 2
    ACTIVE_THEME = "refined"  # Doi thanh "legacy" neu muon quay lai giao dien cu.
    THEMES = {
        "legacy": {
            "BG_COLOR": "#f4efe7",
            "PANEL_COLOR": "#fffaf3",
            "PRIMARY_COLOR": "#1f5c4a",
            "PRIMARY_ACTIVE_BG": "#174336",
            "SECONDARY_COLOR": "#c96c3a",
            "SECONDARY_ACTIVE_BG": "#ab5528",
            "TEXT_COLOR": "#1f2933",
            "MUTED_COLOR": "#6b7280",
            "BORDER_COLOR": "#ded3c4",
            "STATUS_COLOR": "#efe7da",
            "CAMERA_BG": "#111111",
            "TITLE_FONT": ("Segoe UI", 18, "bold"),
            "SECTION_FONT": ("Segoe UI", 16, "bold"),
            "BUTTON_FONT": ("Segoe UI", 18, "bold"),
            "BODY_FONT": ("Segoe UI", 10),
            "SMALL_FONT": ("Segoe UI", 10),
            "BADGE_BG": "#1f5c4a",
            "DANGER_COLOR": "#dc2626",
            "NEUTRAL_BUTTON_BG": "#4b5563",
            "NEUTRAL_BUTTON_ACTIVE_BG": "#374151",
            "SOFT_BUTTON_BG": "#e7ddd0",
            "SOFT_BUTTON_ACTIVE_BG": "#d8cab9",
            "LIGHT_PANEL_ACTIVE_BG": "#f0e6d8",
            "DISABLED_PRIMARY_BG": "#9aa5a1",
            "DISABLED_SECONDARY_BG": "#c7a08f",
            "DISABLED_NEUTRAL_BG": "#9aa5a1",
            "HEADER_ACCENT": "#1f5c4a",
            "INFO_ACCENT": "#3b82f6",
            "WARNING_ACCENT": "#92400e",
            "TREE_HEADING_BG": "#efe7da",
            "TREE_HEADING_FG": "#1f2933",
            "TREE_ROW_BG": "#fffaf3",
            "TREE_SELECTED_BG": "#dce8e3",
            "ENTRY_BG": "#ffffff",
            "ENTRY_BORDER": "#cbbda9",
            "SUBTLE_PANEL": "#f7f1e8",
            "ACTION_CARD_BG": "#fdf8f1",
            "ACTION_CARD_BORDER": "#d8cab8",
            "STATUS_OK_BG": "#edf5f1",
            "STATUS_WARN_BG": "#fbefe7",
            "SIDEBAR_COLOR": "#1A1A1B",
            "SIDEBAR_MUTED": "#9CA3AF",
            "SIDEBAR_ACTIVE": "#242426",
        },
        "refined": {
            "BG_COLOR": "#F0F2F5",
            "PANEL_COLOR": "#FFFFFF",
            "PRIMARY_COLOR": "#4CAF50",
            "PRIMARY_ACTIVE_BG": "#3E8E41",
            "SECONDARY_COLOR": "#2F80ED",
            "SECONDARY_ACTIVE_BG": "#2567BE",
            "TEXT_COLOR": "#1F2937",
            "MUTED_COLOR": "#6B7280",
            "BORDER_COLOR": "#D8DEE8",
            "STATUS_COLOR": "#F8FAFC",
            "CAMERA_BG": "#0B0F14",
            "TITLE_FONT": ("Segoe UI Semibold", 20, "bold"),
            "SECTION_FONT": ("Segoe UI Semibold", 16, "bold"),
            "BUTTON_FONT": ("Segoe UI Semibold", 18, "bold"),
            "BODY_FONT": ("Segoe UI", 10),
            "SMALL_FONT": ("Segoe UI", 10),
            "BADGE_BG": "#4CAF50",
            "DANGER_COLOR": "#E53935",
            "NEUTRAL_BUTTON_BG": "#1A1A1B",
            "NEUTRAL_BUTTON_ACTIVE_BG": "#2A2A2C",
            "SOFT_BUTTON_BG": "#E6EAF0",
            "SOFT_BUTTON_ACTIVE_BG": "#D7DDE7",
            "LIGHT_PANEL_ACTIVE_BG": "#F8FAFC",
            "DISABLED_PRIMARY_BG": "#A5B4AC",
            "DISABLED_SECONDARY_BG": "#9FB6D8",
            "DISABLED_NEUTRAL_BG": "#9CA3AF",
            "HEADER_ACCENT": "#4CAF50",
            "INFO_ACCENT": "#2F80ED",
            "WARNING_ACCENT": "#B45309",
            "TREE_HEADING_BG": "#EEF2F7",
            "TREE_HEADING_FG": "#1F2937",
            "TREE_ROW_BG": "#FFFFFF",
            "TREE_SELECTED_BG": "#DCEBFF",
            "ENTRY_BG": "#FFFFFF",
            "ENTRY_BORDER": "#CBD5E1",
            "SUBTLE_PANEL": "#F8FAFC",
            "ACTION_CARD_BG": "#FFFFFF",
            "ACTION_CARD_BORDER": "#D8DEE8",
            "STATUS_OK_BG": "#EAF7EE",
            "STATUS_WARN_BG": "#FFF4E5",
            "SIDEBAR_COLOR": "#1A1A1B",
            "SIDEBAR_MUTED": "#9CA3AF",
            "SIDEBAR_ACTIVE": "#242426",
        },
    }
    BG_COLOR = "#F0F2F5"
    PANEL_COLOR = "#FFFFFF"
    PRIMARY_COLOR = "#4CAF50"
    SECONDARY_COLOR = "#2F80ED"
    TEXT_COLOR = "#1F2937"
    MUTED_COLOR = "#6B7280"
    BORDER_COLOR = "#D8DEE8"
    STATUS_COLOR = "#F8FAFC"
    SIDEBAR_COLOR = "#1A1A1B"
    WINDOW_WIDTH = 960
    WINDOW_HEIGHT = 640
    MIN_WINDOW_WIDTH = 430
    MIN_WINDOW_HEIGHT = 540
    CAMERA_BG = "#0B0F14"
    CAMERA_PANEL_WIDTH = 380
    CAMERA_PANEL_HEIGHT = 430
    CAMERA_PORTRAIT_ASPECT = 3 / 4
    CAMERA_DISPLAY_MIRROR = _get_bool_env("LOCKER_CAMERA_DISPLAY_MIRROR", True)
    EVENT_LOG_FILE = "locker_events.csv"
    MAX_EVENT_LOG_ROWS = 40
    MAX_EVENT_SNAPSHOTS = 40
    ADMIN_THUMBNAIL_SIZE = (72, 72)
    ADMIN_PASSWORD = "Tuoc1234"
    ADMIN_PIN = os.getenv("LOCKER_ADMIN_PIN", "123456")
    SNAPSHOT_DIR = "event_snapshots"
    FAILED_ALERT_THRESHOLD = 3
    DUPLICATE_FACE_DISTANCE_THRESHOLD = 0.36
    DUPLICATE_FACE_MIN_MATCHES = 4
    DUPLICATE_FACE_MATCH_RATIO = 0.35
    DEFAULT_ITEM_PROFILE = {
        "label": "Vat dung gui",
        "recommended_lockers": (),
    }

    @staticmethod
    def _format_backend_text(result):
        return ""

    @staticmethod
    def _status_after_action(action):
        status_map = {
            "keep_cancelled": "Khong doi",
            "keep_blocked": "Dang su dung",
            "keep_rejected": "Khong doi",
            "register_failed": "Khong doi",
            "register_success": "Dang su dung",
            "take_cancelled": "Khong doi",
            "take_rejected": "Khong doi",
            "take_temporary": "Dang su dung",
            "take_finish": "Con trong",
            "admin_force_clear": "Con trong",
            "admin_clear_fingerprints": "Khong doi",
        }
        return status_map.get(action, "Khong ro")

    @staticmethod
    def _status_from_history_label(action_label):
        status_map = {
            "Huy gui do": "Khong doi",
            "Chan gui do": "Dang su dung",
            "Tu choi gui do": "Khong doi",
            "Dang ky that bai": "Khong doi",
            "Dang ky thanh cong": "Dang su dung",
            "Huy lay do": "Khong doi",
            "Tu choi lay do": "Khong doi",
            "Lay do tam thoi": "Dang su dung",
            "Ket thuc gui do": "Con trong",
            "Admin ep xoa": "Con trong",
            "Admin xoa toan bo van tay": "Khong doi",
        }
        return status_map.get(action_label, "Khong ro")

    @staticmethod
    def _alert_level_from_history_label(action_label):
        return "Canh bao" if action_label == "Tu choi lay do" else "Khong"

    @classmethod
    def _get_take_confidence_threshold(cls, result):
        backend_text = str(result.get("backend", "")).lower()
        if "masked" in backend_text or "mask" in backend_text:
            return cls.TAKE_MASKED_CONFIDENCE_THRESHOLD
        return cls.TAKE_CONFIDENCE_THRESHOLD

    @classmethod
    def _get_take_review_confidence_threshold(cls, result):
        backend_text = str(result.get("backend", "")).lower()
        if "masked" in backend_text or "mask" in backend_text:
            return cls.TAKE_REVIEW_MASKED_CONFIDENCE_THRESHOLD
        return cls.TAKE_REVIEW_CONFIDENCE_THRESHOLD

    @classmethod
    def _get_take_recheck_accept_confidence_threshold(cls, result):
        backend_text = str(result.get("backend", "")).lower()
        if "masked" in backend_text or "mask" in backend_text:
            return cls.TAKE_RECHECK_ACCEPT_MASKED_CONFIDENCE_THRESHOLD
        return cls.TAKE_RECHECK_ACCEPT_CONFIDENCE_THRESHOLD

    def _append_event_log(self, action, result, note):
        action_labels = {
            "keep_cancelled": "Huy gui do",
            "keep_blocked": "Chan gui do",
            "keep_rejected": "Tu choi gui do",
            "register_failed": "Dang ky that bai",
            "register_success": "Dang ky thanh cong",
            "take_cancelled": "Huy lay do",
            "take_rejected": "Tu choi lay do",
            "take_temporary": "Lay do tam thoi",
            "take_finish": "Ket thuc gui do",
            "admin_force_clear": "Admin ep xoa",
            "admin_clear_fingerprints": "Admin xoa toan bo van tay",
        }
        header = [
            "thoi_gian",
            "hanh_dong",
            "ten_tu",
            "do_tin_cay",
            "co_che_xac_thuc",
            "trang_thai_sau",
            "canh_bao",
            "anh_su_kien",
            "ghi_chu",
        ]
        rows = []
        if os.path.exists(self.EVENT_LOG_FILE):
            try:
                with open(self.EVENT_LOG_FILE, "r", newline="", encoding="utf-8") as file_obj:
                    rows = [
                        [row.get(column, "") for column in header]
                        for row in csv.DictReader(file_obj)
                    ]
            except Exception:
                rows = []

        try:
            snapshot_path = self._save_event_snapshot(action, result)
            alert_text = self._register_security_signal(action)
            rows.append(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    action_labels.get(action, action),
                    result.get("name", ""),
                    f"{result.get('confidence', 0.0):.1f}",
                    result.get("backend", ""),
                    self._status_after_action(action),
                    alert_text,
                    snapshot_path,
                    note,
                ]
            )
            rows = rows[-self.MAX_EVENT_LOG_ROWS:]

            with open(self.EVENT_LOG_FILE, "w", newline="", encoding="utf-8") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(header)
                writer.writerows(rows)
            self._cleanup_unreferenced_event_snapshots(rows, header)
        finally:
            self.pending_event_snapshot_frame = None

    def __init__(self, root):
        self.root = root
        self._apply_theme()
        self.root.title("Tu do nhan dien khuon mat")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        usable_width = max(1, screen_width)
        usable_height = max(1, screen_height)
        self.window_width_px = usable_width
        self.window_height_px = usable_height
        self.root.geometry(f"{self.window_width_px}x{self.window_height_px}+0+0")
        self.root.minsize(min(self.MIN_WINDOW_WIDTH, self.window_width_px), min(self.MIN_WINDOW_HEIGHT, self.window_height_px))
        self.root.resizable(True, True)
        self.header_wrap_px = max(220, self.window_width_px - 210)
        self.content_wrap_px = max(260, self.window_width_px - 100)
        self.card_wrap_px = max(150, (self.window_width_px - 90) // 2)
        self.status_wrap_px = max(160, (self.window_width_px - 90) // 2)
        if _get_bool_env("LOCKER_FULLSCREEN", False):
            self.root.attributes("-fullscreen", True)
        elif screen_height >= screen_width:
            self.root.attributes("-fullscreen", _get_bool_env("LOCKER_PORTRAIT_FULLSCREEN", True))
        else:
            try:
                self.root.state("zoomed")
            except tk.TclError:
                pass
        self.root.configure(bg=self.BG_COLOR)
        self._setup_fonts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.option_add("*Font", self.BODY_FONT)
        self.root.option_add("*Entry.Background", self.ENTRY_BG)
        self.root.option_add("*Entry.Relief", "flat")
        self.root.option_add("*Entry.HighlightThickness", 1)

        self.status_var = tk.StringVar(value="San sang.")
        self.available_var = tk.StringVar()
        self.available_note_var = tk.StringVar(value="Dang cap nhat trang thai tu.")
        self.camera_title_var = tk.StringVar(value="")
        self.camera_status_var = tk.StringVar(value="")
        self.security_alert_var = tk.StringVar(value="Khong co canh bao.")
        self.auth_mode_var = tk.StringVar(
            value=self._normalize_auth_mode(
                os.getenv("LOCKER_AUTH_MODE", self.AUTH_MODE_FACE_AND_FINGERPRINT)
            )
        )
        self.fingerprint_status_var = tk.StringVar(value="")
        self.fingerprint_title_var = tk.StringVar(value="")
        self.fingerprint_prompt_var = tk.StringVar(value="")
        self.fingerprint_screen_status_var = tk.StringVar(value="")
        self.door_title_var = tk.StringVar(value="")
        self.door_status_var = tk.StringVar(value="")

        self.camera_cap = None
        self.camera_after_id = None
        self.camera_photo = None
        self.camera_session = None
        self.camera_callback = None
        self.camera_cancel_result = None
        self.camera_close_before_complete = False
        self.admin_preview_images = {}
        self.latest_camera_frame = None
        self.pending_event_snapshot_frame = None
        self.failed_auth_attempts = 0
        self.pending_item_profile = None
        self.pending_locker_id = None
        self.pending_mask_capture = False
        self.pending_glasses_capture = False
        self.pending_used_mask_samples = False
        self.pending_used_glasses_samples = False
        self.last_health_warning = ""
        self.is_action_busy = False
        self.is_training_model = False
        self.active_door_locker_id = None
        self.door_auto_return_after_id = None
        self.door_countdown_remaining = 0
        self.fingerprint_retry_after_id = None
        self.fingerprint_retry_deadline = 0.0
        self.fingerprint_retry_in_progress = False
        self.fingerprint_reconnect_lock = threading.RLock()
        self.admin_panel_frame = None
        self.admin_login_key_bind_id = None
        self.mode3_terminal_last_packet = ""
        self.mode3_terminal_last_result = ""
        self.relay_controller = RelayController(
            locker_pins=self.RELAY_PIN_MAP,
            active_low=True,
            door_sensor_pins=self.DOOR_SENSOR_PIN_MAP,
            door_sensor_closed_active_low=self.DOOR_SENSOR_CLOSED_ACTIVE_LOW,
            stop_relay_when_door_opens=self.STOP_RELAY_WHEN_DOOR_OPENS,
            min_pulse_seconds_before_open_check=self.RELAY_MIN_ON_SECONDS_BEFORE_OPEN_CHECK,
            door_open_settle_seconds=self.DOOR_OPEN_SIGNAL_SETTLE_SECONDS,
            pulse_poll_seconds=self.RELAY_PULSE_POLL_SECONDS,
        )
        self.fingerprint_controller = FingerprintController()
        self.ttk_style = ttk.Style()
        self._configure_ttk_styles()
        self.logo_photo = self._create_university_logo()

        self.setup_ui()

        self.keep_flow_window_title = "Giu do - Nhan dien va dang ky"
        self._update_auth_mode_status()
        self.refresh_availability_label()
        self._start_fingerprint_startup_retry()

    def _place_child_window(self, window, width, height, parent_window=None, modal=True):
        parent = parent_window or self.root
        parent.update_idletasks()
        window.update_idletasks()

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        width = min(width, screen_width)
        height = min(height, screen_height)

        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = max(parent.winfo_width(), 1)
        parent_height = max(parent.winfo_height(), 1)

        x = parent_x + max(0, (parent_width - width) // 2)
        y = parent_y + max(0, (parent_height - height) // 2)
        x = max(0, min(x, max(0, screen_width - width)))
        y = max(0, min(y, max(0, screen_height - height)))

        window.geometry(f"{width}x{height}+{x}+{y}")
        window.transient(parent)
        if modal:
            window.grab_set()
        window.lift(parent)
        window.focus_force()

    def _apply_theme(self):
        theme = self.THEMES.get(self.ACTIVE_THEME, self.THEMES["legacy"])
        for key, value in theme.items():
            setattr(self, key, value)

    def _setup_fonts(self):
        available_families = {family.lower(): family for family in tkfont.families(self.root)}
        base_family = available_families.get("segoe ui", "Helvetica")
        semibold_family = available_families.get("segoe ui semibold", base_family)

        self.TITLE_FONT = tkfont.Font(family=semibold_family, size=20, weight="bold")
        self.SECTION_FONT = tkfont.Font(family=semibold_family, size=15, weight="bold")
        self.BUTTON_FONT = tkfont.Font(family=semibold_family, size=13, weight="bold")
        self.BODY_FONT = tkfont.Font(family=base_family, size=10)
        self.SMALL_FONT = tkfont.Font(family=base_family, size=9)
        self.MICRO_FONT = tkfont.Font(family=base_family, size=8)
        self.SIDEBAR_TITLE_FONT = tkfont.Font(family=semibold_family, size=13, weight="bold")
        self.HERO_FONT = tkfont.Font(family=semibold_family, size=22, weight="bold")
        self.STATUS_FONT = tkfont.Font(family=semibold_family, size=10, weight="bold")
        self.STATUS_ICON_FONT = tkfont.Font(family=base_family, size=15, weight="bold")
        self.LARGE_ICON_FONT = tkfont.Font(family=base_family, size=42, weight="bold")
        self.CANVAS_BUTTON_SUBFONT = tkfont.Font(family=base_family, size=8)

    def _configure_ttk_styles(self):
        self.ttk_style.theme_use("clam")
        self.ttk_style.configure(
            "Treeview",
            background=self.TREE_ROW_BG,
            fieldbackground=self.TREE_ROW_BG,
            foreground=self.TEXT_COLOR,
            rowheight=28,
            borderwidth=0,
            relief="flat",
            font=self.BODY_FONT,
        )
        self.ttk_style.configure(
            "Treeview.Heading",
            background=self.TREE_HEADING_BG,
            foreground=self.TREE_HEADING_FG,
            borderwidth=0,
            relief="flat",
            font=("Segoe UI Semibold", 10, "bold"),
            padding=(8, 8),
        )
        self.ttk_style.map(
            "Treeview",
            background=[("selected", self.TREE_SELECTED_BG)],
            foreground=[("selected", self.TEXT_COLOR)],
        )
        self.ttk_style.map(
            "Treeview.Heading",
            background=[("active", self.TREE_HEADING_BG)],
            relief=[("active", "flat")],
        )
        self.ttk_style.configure(
            "Vertical.TScrollbar",
            background=self.TREE_HEADING_BG,
            troughcolor=self.STATUS_COLOR,
            bordercolor=self.STATUS_COLOR,
            arrowcolor=self.TEXT_COLOR,
        )
        self.ttk_style.configure(
            "Horizontal.TScrollbar",
            background=self.TREE_HEADING_BG,
            troughcolor=self.STATUS_COLOR,
            bordercolor=self.STATUS_COLOR,
            arrowcolor=self.TEXT_COLOR,
        )

    def _create_university_logo(self, size=(64, 64)):
        logo_path = os.path.join(os.path.dirname(__file__), "logodaihocbachkhoa.jpg")
        try:
            image = Image.open(logo_path).convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", size, "white")
            offset_x = (size[0] - image.width) // 2
            offset_y = (size[1] - image.height) // 2
            canvas.paste(image, (offset_x, offset_y))
            return ImageTk.PhotoImage(canvas)
        except Exception:
            fallback = Image.new("RGB", size, "#fffdfa")
            return ImageTk.PhotoImage(fallback)

    def setup_ui(self):
        self.sidebar_width_px = 132 if self.window_width_px < 720 else 186
        self.content_wrap_px = max(210, self.window_width_px - self.sidebar_width_px - 92)
        self.card_wrap_px = max(170, self.content_wrap_px - 42)
        self.status_wrap_px = max(170, self.content_wrap_px - 36)

        self.main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, minsize=self.sidebar_width_px)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self._build_sidebar(self.main_frame)
        self._build_main_content(self.main_frame)
        self._build_camera_screen()
        self._build_fingerprint_screen()
        self._build_door_screen()

    def _make_panel(self, master, bg=None, padx=16, pady=16):
        return tk.Frame(
            master,
            bg=bg or self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=padx,
            pady=pady,
        )

    def _make_scrollable_page(self, master, bg=None, padx=0, pady=0):
        page_bg = bg or self.BG_COLOR
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(master, bg=page_bg, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(master, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        page = tk.Frame(canvas, bg=page_bg, padx=padx, pady=pady)
        page_id = canvas.create_window((0, 0), window=page, anchor="nw")

        def update_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def fit_page_width(event):
            canvas.itemconfigure(page_id, width=event.width)

        def scroll_with_wheel(event):
            delta = 0
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            elif getattr(event, "delta", 0):
                delta = -int(event.delta / 120)
            if delta:
                try:
                    canvas.yview_scroll(delta, "units")
                except tk.TclError:
                    unbind_wheel()

        def bind_wheel(_event=None):
            canvas.bind_all("<MouseWheel>", scroll_with_wheel)
            canvas.bind_all("<Button-4>", scroll_with_wheel)
            canvas.bind_all("<Button-5>", scroll_with_wheel)

        def unbind_wheel(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        page.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", fit_page_width)
        canvas.bind("<Enter>", bind_wheel)
        page.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)
        page.bind("<Leave>", unbind_wheel)
        master.bind("<Destroy>", lambda event: unbind_wheel() if event.widget is master else None, add="+")
        return page

    def _make_canvas_button(
        self,
        master,
        text,
        command,
        fill,
        active_fill=None,
        disabled_fill=None,
        parent_bg=None,
        width=220,
        height=58,
        subtext="",
        icon="",
        fg="white",
    ):
        return CanvasButton(
            master,
            text=text,
            command=command,
            fill=fill,
            fg=fg,
            active_fill=active_fill,
            disabled_fill=disabled_fill or self.DISABLED_NEUTRAL_BG,
            parent_bg=parent_bg or master.cget("bg"),
            width=width,
            height=height,
            radius=18,
            font=self.BUTTON_FONT,
            subtext=subtext,
            subfont=self.CANVAS_BUTTON_SUBFONT,
            icon=icon,
            icon_font=self.SECTION_FONT,
        )

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=self.SIDEBAR_COLOR, padx=16, pady=20)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.configure(width=self.sidebar_width_px)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(5, weight=1)

        tk.Label(sidebar, image=self.logo_photo, bg=self.SIDEBAR_COLOR).grid(row=0, column=0, sticky="w")

        tk.Label(
            sidebar,
            text="SMART\nLOCKER",
            font=self.SIDEBAR_TITLE_FONT,
            fg="white",
            bg=self.SIDEBAR_COLOR,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(16, 6))

        tk.Label(
            sidebar,
            text="Face ID kiosk",
            font=self.SMALL_FONT,
            fg=self.SIDEBAR_MUTED,
            bg=self.SIDEBAR_COLOR,
            justify="left",
            wraplength=max(92, self.sidebar_width_px - 34),
        ).grid(row=2, column=0, sticky="w")

        nav_items = (("FACE", "Nhan dien"), ("FINGER", "Van tay"), ("RELAY", "Mo khoa"))
        nav_frame = tk.Frame(sidebar, bg=self.SIDEBAR_COLOR)
        nav_frame.grid(row=3, column=0, sticky="ew", pady=(28, 0))
        for label, detail in nav_items:
            row = tk.Frame(nav_frame, bg=self.SIDEBAR_ACTIVE, padx=10, pady=8)
            row.pack(fill="x", pady=(0, 8))
            tk.Label(row, text=label, font=self.MICRO_FONT, fg=self.PRIMARY_COLOR, bg=self.SIDEBAR_ACTIVE).pack(anchor="w")
            tk.Label(row, text=detail, font=self.SMALL_FONT, fg="white", bg=self.SIDEBAR_ACTIVE).pack(anchor="w")

        self.admin_button = self._make_canvas_button(
            sidebar,
            text="Quan tri",
            command=self.show_admin_login,
            fill=self.NEUTRAL_BUTTON_BG,
            active_fill=self.NEUTRAL_BUTTON_ACTIVE_BG,
            disabled_fill=self.DISABLED_NEUTRAL_BG,
            parent_bg=self.SIDEBAR_COLOR,
            width=92 if self.sidebar_width_px < 150 else 108,
            height=36,
        )
        self.admin_button.grid(row=6, column=0, sticky="w", pady=(18, 0))

    def _build_main_content(self, parent):
        content = tk.Frame(parent, bg=self.BG_COLOR, padx=22, pady=22)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(3, weight=1)

        header = tk.Frame(content, bg=self.BG_COLOR)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="Tu luu tru thong minh",
            font=self.HERO_FONT,
            fg=self.TEXT_COLOR,
            bg=self.BG_COLOR,
            anchor="w",
            justify="left",
            wraplength=self.content_wrap_px,
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            header,
            text="Gui do va lay do bang nhan dien khuon mat, van tay va relay tu dong.",
            font=self.BODY_FONT,
            fg=self.MUTED_COLOR,
            bg=self.BG_COLOR,
            anchor="w",
            justify="left",
            wraplength=self.content_wrap_px,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        availability_panel = self._make_panel(content, padx=18, pady=16)
        availability_panel.grid(row=1, column=0, sticky="ew", pady=(20, 18))
        availability_panel.grid_columnconfigure(0, weight=1)

        tk.Label(
            availability_panel,
            text="Trang thai ngan tu",
            font=self.SECTION_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).grid(row=0, column=0, sticky="w")

        self.availability_badge = tk.Label(
            availability_panel,
            textvariable=self.available_var,
            font=self.STATUS_FONT,
            fg="white",
            bg=self.PRIMARY_COLOR,
            padx=14,
            pady=7,
        )
        self.availability_badge.grid(row=0, column=1, sticky="e", padx=(16, 0))

        self.locker_status_canvas = tk.Canvas(
            availability_panel,
            height=78,
            bg=self.PANEL_COLOR,
            bd=0,
            highlightthickness=0,
        )
        self.locker_status_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 8))
        self.locker_status_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_locker_status(self._get_used_locker_ids()),
        )

        tk.Label(
            availability_panel,
            textvariable=self.available_note_var,
            font=self.SMALL_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        action_frame = tk.Frame(content, bg=self.BG_COLOR)
        action_frame.grid(row=2, column=0, sticky="ew")
        action_frame.grid_columnconfigure(0, weight=1)
        if self.window_width_px >= 760:
            action_frame.grid_columnconfigure(1, weight=1)

        self.keep_button = self._make_canvas_button(
            action_frame,
            text="GUI DO",
            subtext="Dang ky va cap tu trong",
            command=self.handle_keep_item,
            fill=self.PRIMARY_COLOR,
            active_fill=self.PRIMARY_ACTIVE_BG,
            disabled_fill=self.DISABLED_PRIMARY_BG,
            parent_bg=self.BG_COLOR,
            height=88,
            icon="+",
        )
        self.take_button = self._make_canvas_button(
            action_frame,
            text="LAY DO",
            subtext="Xac minh chu so huu",
            command=self.show_take_options,
            fill=self.SECONDARY_COLOR,
            active_fill=self.SECONDARY_ACTIVE_BG,
            disabled_fill=self.DISABLED_SECONDARY_BG,
            parent_bg=self.BG_COLOR,
            height=88,
            icon="\u21E3",
        )

        if self.window_width_px >= 760:
            self.keep_button.grid(row=0, column=0, sticky="ew", padx=(0, 10))
            self.take_button.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        else:
            self.keep_button.grid(row=0, column=0, sticky="ew", pady=(0, 12))
            self.take_button.grid(row=1, column=0, sticky="ew")

        lower_area = tk.Frame(content, bg=self.BG_COLOR)
        lower_area.grid(row=3, column=0, sticky="nsew", pady=(18, 0))
        lower_area.grid_columnconfigure(0, weight=1)
        lower_area.grid_columnconfigure(1, weight=1)

        security_bar = self._make_panel(lower_area, bg=self.STATUS_WARN_BG, padx=14, pady=12)
        status_bar = self._make_panel(lower_area, bg=self.STATUS_OK_BG, padx=14, pady=12)
        if self.window_width_px >= 760:
            security_bar.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
            status_bar.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        else:
            security_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
            status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        tk.Label(
            security_bar,
            text="CANH BAO",
            font=self.STATUS_FONT,
            fg=self.WARNING_ACCENT,
            bg=self.STATUS_WARN_BG,
        ).pack(anchor="w")
        tk.Label(
            security_bar,
            textvariable=self.security_alert_var,
            font=self.SMALL_FONT,
            fg=self.TEXT_COLOR,
            bg=self.STATUS_WARN_BG,
            wraplength=self.status_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(7, 0))

        tk.Label(
            status_bar,
            text="HE THONG",
            font=self.STATUS_FONT,
            fg=self.PRIMARY_COLOR,
            bg=self.STATUS_OK_BG,
        ).pack(anchor="w")
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            font=self.SMALL_FONT,
            fg=self.TEXT_COLOR,
            bg=self.STATUS_OK_BG,
            wraplength=self.status_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(7, 0))
        tk.Label(
            status_bar,
            textvariable=self.fingerprint_status_var,
            font=self.MICRO_FONT,
            fg=self.MUTED_COLOR,
            bg=self.STATUS_OK_BG,
            wraplength=self.status_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

    def _build_camera_screen(self):
        self.camera_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=18, pady=18)
        self.camera_frame.grid_columnconfigure(0, weight=1)
        self.camera_frame.grid_rowconfigure(1, weight=1)

        camera_header = self._make_panel(self.camera_frame, padx=18, pady=14)
        camera_header.grid(row=0, column=0, sticky="ew")
        camera_header.grid_columnconfigure(0, weight=1)

        tk.Label(
            camera_header,
            textvariable=self.camera_title_var,
            font=self.SECTION_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            camera_header,
            text="Can khuon mat nam trong vung oval sang de he thong quet on dinh.",
            font=self.SMALL_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.camera_preview_wrap = tk.Frame(
            self.camera_frame,
            bg=self.CAMERA_BG,
            highlightbackground="#151C26",
            highlightthickness=1,
        )
        self.camera_preview_wrap.grid(row=1, column=0, sticky="nsew", pady=(14, 12))

        self.camera_preview = tk.Label(self.camera_preview_wrap, bg=self.CAMERA_BG, relief="flat")
        self.camera_preview.pack(fill="both", expand=True)

        camera_status_card = self._make_panel(self.camera_frame, bg=self.PANEL_COLOR, padx=14, pady=12)
        camera_status_card.grid(row=2, column=0, sticky="ew")
        tk.Label(
            camera_status_card,
            textvariable=self.camera_status_var,
            font=self.SMALL_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w")

        self.camera_cancel_button = self._make_canvas_button(
            self.camera_frame,
            text="HUY",
            command=self.cancel_camera_stream,
            fill=self.SOFT_BUTTON_BG,
            active_fill=self.SOFT_BUTTON_ACTIVE_BG,
            disabled_fill=self.DISABLED_NEUTRAL_BG,
            parent_bg=self.BG_COLOR,
            height=54,
            fg=self.TEXT_COLOR,
        )
        self.camera_cancel_button.grid(row=3, column=0, sticky="ew", pady=(12, 0))

    def _build_fingerprint_screen(self):
        self.fingerprint_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=22, pady=22)

        fingerprint_header = self._make_panel(self.fingerprint_frame, padx=18, pady=16)
        fingerprint_header.pack(fill="x")

        tk.Label(
            fingerprint_header,
            textvariable=self.fingerprint_title_var,
            font=self.SECTION_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")
        tk.Label(
            fingerprint_header,
            textvariable=self.fingerprint_prompt_var,
            font=self.BODY_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        fingerprint_card = self._make_panel(self.fingerprint_frame, bg=self.PANEL_COLOR, padx=22, pady=34)
        fingerprint_card.pack(fill="both", expand=True, pady=(16, 0))

        tk.Label(
            fingerprint_card,
            text="\u25CE",
            font=self.LARGE_ICON_FONT,
            fg=self.PRIMARY_COLOR,
            bg=self.PANEL_COLOR,
        ).pack()
        tk.Label(
            fingerprint_card,
            text="VAN TAY",
            font=self.HERO_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(pady=(4, 0))

        self.fingerprint_progress = ttk.Progressbar(fingerprint_card, mode="indeterminate", length=260)
        self.fingerprint_progress.pack(pady=(22, 14))

        tk.Label(
            fingerprint_card,
            textvariable=self.fingerprint_screen_status_var,
            font=self.BODY_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="center",
        ).pack()

    def _build_door_screen(self):
        self.door_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=22, pady=22)

        door_header = self._make_panel(self.door_frame, padx=18, pady=16)
        door_header.pack(fill="x")

        tk.Label(
            door_header,
            textvariable=self.door_title_var,
            font=self.SECTION_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")
        tk.Label(
            door_header,
            text="Hay dong cua tu sau khi thao tac xong.",
            font=self.BODY_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        door_card = self._make_panel(self.door_frame, bg=self.PANEL_COLOR, padx=22, pady=34)
        door_card.pack(fill="both", expand=True, pady=(16, 0))
        tk.Label(
            door_card,
            text="\U0001F512",
            font=self.LARGE_ICON_FONT,
            fg=self.PRIMARY_COLOR,
            bg=self.PANEL_COLOR,
        ).pack()
        tk.Label(
            door_card,
            textvariable=self.door_status_var,
            font=self.SECTION_FONT,
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="center",
        ).pack(expand=True, pady=(14, 0))

        self.door_button_frame = tk.Frame(self.door_frame, bg=self.BG_COLOR)
        self.door_button_frame.pack(fill="x", pady=(14, 0))
        self.door_button_frame.grid_columnconfigure(0, weight=1)
        self.door_button_frame.grid_columnconfigure(1, weight=1)

        self.reopen_door_button = self._make_canvas_button(
            self.door_button_frame,
            text="MO LAI TU",
            command=self._reopen_active_locker_from_door_screen,
            fill=self.PRIMARY_COLOR,
            active_fill=self.PRIMARY_ACTIVE_BG,
            disabled_fill=self.DISABLED_PRIMARY_BG,
            parent_bg=self.BG_COLOR,
            height=56,
        )
        self.return_from_door_button = self._make_canvas_button(
            self.door_button_frame,
            text="TRO LAI",
            command=self._return_from_door_screen,
            fill=self.SOFT_BUTTON_BG,
            active_fill=self.SOFT_BUTTON_ACTIVE_BG,
            disabled_fill=self.DISABLED_NEUTRAL_BG,
            parent_bg=self.BG_COLOR,
            height=56,
            fg=self.TEXT_COLOR,
        )
        self.reopen_door_button.grid(row=0, column=0, padx=(0, 7), sticky="ew")
        self.return_from_door_button.grid(row=0, column=1, padx=(7, 0), sticky="ew")
        self.door_button_frame.pack_forget()

    def _draw_locker_status(self, used_ids):
        canvas = getattr(self, "locker_status_canvas", None)
        if canvas is None:
            return

        canvas.delete("all")
        width = max(220, canvas.winfo_width())
        height = max(70, canvas.winfo_height())
        left = 28
        right = width - 28
        center_y = 31
        radius = 18
        count = max(1, len(self.LOCKER_IDS))
        spacing = (right - left) / max(1, count - 1)

        canvas.create_line(
            left,
            center_y,
            right,
            center_y,
            fill="#E5E7EB",
            width=7,
            capstyle=tk.ROUND,
        )

        used_ids = set(used_ids or [])
        for index, locker_id in enumerate(self.LOCKER_IDS):
            x = left + spacing * index
            occupied = locker_id in used_ids
            fill = self.DANGER_COLOR if occupied else self.PANEL_COLOR
            outline = self.DANGER_COLOR if occupied else self.PRIMARY_COLOR
            canvas.create_oval(
                x - radius,
                center_y - radius,
                x + radius,
                center_y + radius,
                fill=fill,
                outline=outline,
                width=3,
            )
            canvas.create_text(
                x,
                center_y,
                text="\U0001F512" if occupied else locker_id,
                fill="white" if occupied else self.PRIMARY_COLOR,
                font=self.STATUS_ICON_FONT if occupied else self.STATUS_FONT,
            )
            canvas.create_text(
                x,
                height - 15,
                text=f"Tu {locker_id}",
                fill=self.MUTED_COLOR,
                font=self.MICRO_FONT,
            )

    def _update_action_controls(self):
        is_disabled = self.is_action_busy or self.is_training_model
        self.keep_button.config(state=tk.DISABLED if is_disabled else tk.NORMAL)
        self.take_button.config(state=tk.DISABLED if is_disabled else tk.NORMAL)
        self.admin_button.config(state=tk.DISABLED if is_disabled else tk.NORMAL)
        self.keep_button.config(
            bg=self.DISABLED_PRIMARY_BG if is_disabled else self.PRIMARY_COLOR,
            cursor="watch" if is_disabled else "hand2",
        )
        self.take_button.config(
            bg=self.DISABLED_SECONDARY_BG if is_disabled else self.SECONDARY_COLOR,
            cursor="watch" if is_disabled else "hand2",
        )
        self.admin_button.config(
            bg=self.DISABLED_NEUTRAL_BG if is_disabled else self.NEUTRAL_BUTTON_BG,
            cursor="watch" if is_disabled else "hand2",
        )

    def _save_event_snapshot(self, action, result):
        snapshot_frame = self.latest_camera_frame
        if snapshot_frame is None:
            snapshot_frame = self.pending_event_snapshot_frame
        if snapshot_frame is None:
            return ""

        os.makedirs(self.SNAPSHOT_DIR, exist_ok=True)
        locker_name = str(result.get("name", "") or self.pending_locker_id or "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{action}_{locker_name}.jpg"
        file_path = os.path.join(self.SNAPSHOT_DIR, file_name)

        try:
            cv2.imwrite(file_path, snapshot_frame)
        except Exception:
            return ""
        self._trim_event_snapshots()
        return file_path

    def _trim_event_snapshots(self):
        if not os.path.isdir(self.SNAPSHOT_DIR):
            return

        snapshots = []
        for file_name in os.listdir(self.SNAPSHOT_DIR):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            file_path = os.path.join(self.SNAPSHOT_DIR, file_name)
            try:
                modified_time = os.path.getmtime(file_path)
            except OSError:
                continue
            snapshots.append((modified_time, file_path))

        snapshots.sort(reverse=True)
        for _modified_time, file_path in snapshots[self.MAX_EVENT_SNAPSHOTS:]:
            try:
                os.remove(file_path)
            except OSError:
                pass

    def _cleanup_unreferenced_event_snapshots(self, rows=None, header=None):
        if not os.path.isdir(self.SNAPSHOT_DIR):
            return

        header = header or [
            "thoi_gian",
            "hanh_dong",
            "ten_tu",
            "do_tin_cay",
            "co_che_xac_thuc",
            "trang_thai_sau",
            "canh_bao",
            "anh_su_kien",
            "ghi_chu",
        ]
        retained_paths = set()
        for row in rows or []:
            if isinstance(row, dict):
                row_data = row
            else:
                row_data = dict(zip(header, row))
            snapshot_path = self._resolve_event_snapshot_path(row_data)
            if snapshot_path:
                retained_paths.add(os.path.normcase(os.path.abspath(snapshot_path)))

        snapshot_dirs = self._snapshot_path_candidates(self.SNAPSHOT_DIR) or [self.SNAPSHOT_DIR]
        visited_dirs = set()
        for snapshot_dir in snapshot_dirs:
            if not os.path.isdir(snapshot_dir):
                continue
            snapshot_dir = os.path.abspath(snapshot_dir)
            normalized_dir = os.path.normcase(snapshot_dir)
            if normalized_dir in visited_dirs:
                continue
            visited_dirs.add(normalized_dir)

            try:
                file_names = os.listdir(snapshot_dir)
            except OSError:
                continue

            for file_name in file_names:
                if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                file_path = os.path.join(snapshot_dir, file_name)
                normalized_path = os.path.normcase(os.path.abspath(file_path))
                if normalized_path in retained_paths:
                    continue
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def _register_security_signal(self, action):
        if action == "take_rejected":
            self.failed_auth_attempts += 1
            if self.failed_auth_attempts >= self.FAILED_ALERT_THRESHOLD:
                alert_text = (
                    f"Canh bao: {self.failed_auth_attempts} lan xac thuc that bai lien tiep "
                    f"luc {datetime.now().strftime('%H:%M:%S')}"
                )
                self.security_alert_var.set(alert_text)
                return alert_text
            self.security_alert_var.set(
                f"Theo doi: {self.failed_auth_attempts} lan xac thuc that bai lien tiep."
            )
            return "Theo doi"

        if action in {"take_temporary", "take_finish", "register_success", "admin_force_clear"}:
            self.failed_auth_attempts = 0
            self.security_alert_var.set("Khong co canh bao.")
        return "Khong"

    def _build_usage_summary(self):
        rows = self._read_event_history()
        summary = {
            "total_events": len(rows),
            "register_success": 0,
            "take_success": 0,
            "failed_auth": 0,
            "admin_actions": 0,
            "active_alert": self.security_alert_var.get(),
        }
        for row in rows:
            action = row.get("hanh_dong", "")
            if action == "Dang ky thanh cong":
                summary["register_success"] += 1
            elif action in {"Lay do tam thoi", "Ket thuc gui do"}:
                summary["take_success"] += 1
            elif action == "Tu choi lay do":
                summary["failed_auth"] += 1
            elif action == "Admin ep xoa":
                summary["admin_actions"] += 1
        return summary

    def set_busy(self, is_busy, status_text):
        self.is_action_busy = is_busy
        self._update_action_controls()
        self.status_var.set(status_text)
        self.root.update()

    def set_training_wait(self, is_waiting, status_text=None):
        self.is_training_model = is_waiting
        self._update_action_controls()
        if status_text is not None:
            self.status_var.set(status_text)
        self.root.update()

    @classmethod
    def _normalize_auth_mode(cls, value):
        normalized = str(value or "").strip().lower()
        if normalized in {"mode3", "pc", "computer", "face_only", "face_only_pc", "no_hardware"}:
            return cls.AUTH_MODE_FACE_ONLY_PC
        if normalized in {"mode2", "either", "face_or_fingerprint"}:
            return cls.AUTH_MODE_FACE_OR_FINGERPRINT
        return cls.AUTH_MODE_FACE_AND_FINGERPRINT

    def _is_dual_auth_mode(self):
        return self.auth_mode_var.get() == self.AUTH_MODE_FACE_AND_FINGERPRINT

    def _is_pc_only_mode(self):
        return self.auth_mode_var.get() == self.AUTH_MODE_FACE_ONLY_PC

    def _handle_auth_mode_changed(self):
        self._update_auth_mode_status()
        if self._is_pc_only_mode():
            self._cancel_fingerprint_startup_retry()
            if not self.is_action_busy:
                self.status_var.set("Da chon Mode 3: chay tren may tinh, bo qua van tay va relay.")
            return

        self._start_fingerprint_startup_retry()

    def _update_auth_mode_status(self):
        if self._is_pc_only_mode():
            self.fingerprint_status_var.set(
                "Mode 3: chi dung khuon mat tren may tinh\n"
                "Bo qua cam bien van tay, relay va cam bien cua."
            )
            return

        auth_label = (
            "Mode 1: can khuon mat va van tay"
            if self._is_dual_auth_mode()
            else "Mode 2: khuon mat hoac van tay"
        )
        self.fingerprint_status_var.set(f"{auth_label}\n{self.fingerprint_controller.get_status_text()}")

    def _start_fingerprint_startup_retry(self):
        if self._is_pc_only_mode():
            self._cancel_fingerprint_startup_retry()
            return
        if self.fingerprint_controller.is_available():
            return

        self.fingerprint_retry_deadline = time.monotonic() + self.FINGERPRINT_STARTUP_RETRY_SECONDS
        if not self.is_action_busy:
            self.status_var.set("Cam bien van tay chua san sang, dang tu ket noi lai...")
        self._schedule_fingerprint_startup_retry(delay_ms=500)

    def _schedule_fingerprint_startup_retry(self, delay_ms=None):
        self._cancel_fingerprint_startup_retry()
        if self._is_pc_only_mode():
            return
        if self.fingerprint_controller.is_available():
            return
        if time.monotonic() >= self.fingerprint_retry_deadline:
            self._update_auth_mode_status()
            return

        self.fingerprint_retry_after_id = self.root.after(
            self.FINGERPRINT_STARTUP_RETRY_INTERVAL_MS if delay_ms is None else delay_ms,
            self._run_fingerprint_startup_retry,
        )

    def _cancel_fingerprint_startup_retry(self):
        if self.fingerprint_retry_after_id is not None:
            try:
                self.root.after_cancel(self.fingerprint_retry_after_id)
            except Exception:
                pass
            self.fingerprint_retry_after_id = None

    def _run_fingerprint_startup_retry(self):
        self.fingerprint_retry_after_id = None
        if self._is_pc_only_mode():
            return
        if self.fingerprint_controller.is_available() or self.fingerprint_retry_in_progress:
            return
        if time.monotonic() >= self.fingerprint_retry_deadline:
            self._update_auth_mode_status()
            return

        self.fingerprint_retry_in_progress = True

        def worker():
            try:
                with self.fingerprint_reconnect_lock:
                    connected = self.fingerprint_controller.reinitialize()
            except Exception:
                connected = False

            self.root.after(0, lambda: self._finish_fingerprint_startup_retry(connected))

        threading.Thread(target=worker, daemon=True).start()

    def _print_mode3_terminal(self, message):
        if not self._is_pc_only_mode():
            return
        timestamp = time.strftime("%H:%M:%S")
        print(f"[Mode3 {timestamp}] {message}", flush=True)

    @staticmethod
    def _render_terminal_table(headers, values):
        widths = [max(len(str(header)), len(str(value))) for header, value in zip(headers, values)]
        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        header_row = "| " + " | ".join(str(header).ljust(width) for header, width in zip(headers, widths)) + " |"
        value_row = "| " + " | ".join(str(value).ljust(width) for value, width in zip(values, widths)) + " |"
        return "\n".join([separator, header_row, separator, value_row, separator])

    def _emit_mode3_camera_packet(self, packet):
        if not self._is_pc_only_mode():
            return
        summary_text = str(packet.get("summary_text", "")).strip()
        status_text = str(packet.get("status_text", "")).strip()
        backend_text = str(packet.get("backend_text", "")).strip()
        diagnostics = packet.get("diagnostics") or {}
        quality = diagnostics.get("quality") or {}
        prediction = diagnostics.get("prediction") or {}
        diagnostics_signature = "|".join(
            [
                str(quality.get("face_class", "")),
                str(quality.get("brightness", "")),
                str(quality.get("sharpness", "")),
                str(quality.get("face_width_ratio", "")),
                str(prediction.get("distance_normalized", "")),
                str(prediction.get("distance_scale", "")),
            ]
        )
        combined = " | ".join(part for part in (summary_text, status_text, backend_text, diagnostics_signature) if part)
        if combined and combined != self.mode3_terminal_last_packet:
            self.mode3_terminal_last_packet = combined
            predicted_name = "None"
            confidence_text = "0.0%"
            confirmed_text = "0"
            state_text = str(quality.get("face_class", "-"))
            brightness = quality.get("brightness")
            sharpness = quality.get("sharpness")
            ratio = quality.get("face_width_ratio")
            aligned_flag = quality.get("aligned")
            moving_flag = quality.get("moving_too_much")
            aligned_text = "-" if aligned_flag is None else ("yes" if aligned_flag else "no")
            moving_text = "-" if moving_flag is None else ("yes" if moving_flag else "no")
            distance_value = prediction.get("distance")
            normalized_distance = prediction.get("distance_normalized")
            distance_scale = prediction.get("distance_scale", "")
            try:
                summary_parts = [part.strip() for part in summary_text.split("|")]
                for part in summary_parts:
                    if part.startswith("Du doan:"):
                        predicted_name = part.split(":", 1)[1].strip()
                    elif part.startswith("Tin cay:"):
                        confidence_text = part.split(":", 1)[1].strip()
                    elif part.startswith("Da xac nhan:"):
                        confirmed_text = part.split(":", 1)[1].strip()
            except Exception:
                pass
            table = self._render_terminal_table(
                ["TYPE", "PRED", "CONF", "CONFIRMED", "STATE", "D_NORM", "LIGHT", "SHARP", "RATIO", "ALIGN", "MOVE", "BACKEND"],
                [
                    "camera",
                    predicted_name,
                    confidence_text,
                    confirmed_text,
                    state_text,
                    (
                        f"{float(normalized_distance):.3f} ({distance_scale})"
                        if normalized_distance is not None
                        else "-"
                    ),
                    f"{float(brightness):.1f}" if brightness is not None else "-",
                    f"{float(sharpness):.1f}" if sharpness is not None else "-",
                    f"{float(ratio):.3f}" if ratio is not None else "-",
                    aligned_text,
                    moving_text,
                    backend_text or "-",
                ],
            )
            status_table = self._render_terminal_table(
                ["TYPE", "STATUS", "D_RAW"],
                [
                    "status",
                    status_text or "-",
                    f"{float(distance_value):.4f}" if distance_value is not None else "-",
                ],
            )
            self._print_mode3_terminal(f"\n{table}\n{status_table}")

    def _emit_mode3_result(self, stage, result, extra_text=""):
        if not self._is_pc_only_mode():
            return
        if isinstance(result, dict):
            payload = result
            name = payload.get("name", "") or "-"
            confidence = float(payload.get("confidence", 0.0))
            backend = payload.get("backend", "") or "-"
            diagnostics = payload.get("diagnostics") or {}
            quality = diagnostics.get("quality") or {}
            prediction = diagnostics.get("prediction") or {}
            state_text = str(quality.get("face_class", "-"))
            distance_text = "-"
            normalized_text = "-"
            if prediction.get("distance") is not None:
                distance_text = f"{float(prediction['distance']):.4f}"
            if prediction.get("distance_normalized") is not None:
                normalized_text = f"{float(prediction['distance_normalized']):.3f}"
            message = self._render_terminal_table(
                ["TYPE", "NAME", "CONF", "STATE", "D_RAW", "D_NORM", "BACKEND", "DETAIL"],
                [stage, name, f"{confidence:.1f}%", state_text, distance_text, normalized_text, backend, extra_text or "-"],
            )
        else:
            message = self._render_terminal_table(
                ["TYPE", "RESULT", "DETAIL"],
                [stage, repr(result), extra_text or "-"],
            )
        if message != self.mode3_terminal_last_result:
            self.mode3_terminal_last_result = message
            self._print_mode3_terminal(f"\n{message}")

    def _discard_pending_capture_data(self, locker_id):
        if not locker_id:
            return
        try:
            delete_user(locker_id)
        except Exception:
            pass

    def _finish_fingerprint_startup_retry(self, connected):
        self.fingerprint_retry_in_progress = False
        self._update_auth_mode_status()
        if self._is_pc_only_mode():
            return

        if connected:
            if not self.is_action_busy:
                self.status_var.set("Da ket noi cam bien van tay.")
            return

        self._schedule_fingerprint_startup_retry()

    def _ensure_fingerprint_available(self):
        if self._is_pc_only_mode():
            return False
        if not self.fingerprint_controller.is_available():
            self.status_var.set("Dang thu ket noi lai cam bien van tay...")
            self.root.update()
            with self.fingerprint_reconnect_lock:
                self.fingerprint_controller.reinitialize()
            self._update_auth_mode_status()
        return self.fingerprint_controller.is_available()

    def _fingerprint_unavailable_reason(self):
        return self.fingerprint_controller.get_status_text()

    def _show_fingerprint_panel(self, title, prompt, status_text):
        self.fingerprint_title_var.set(title)
        self.fingerprint_prompt_var.set(prompt)
        self.fingerprint_screen_status_var.set(status_text)
        self.main_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.door_frame.pack_forget()
        self._hide_admin_panel()
        if not self.fingerprint_frame.winfo_ismapped():
            self.fingerprint_frame.pack(fill="both", expand=True)
        self.fingerprint_progress.start(12)
        self.root.update()

    def _hide_fingerprint_panel(self):
        self.fingerprint_progress.stop()
        self.fingerprint_frame.pack_forget()
        if not self.main_frame.winfo_ismapped() and not self.door_frame.winfo_ismapped():
            self.main_frame.pack(fill="both", expand=True)
        self.root.update()

    def _update_task_status(self, status_text):
        self.status_var.set(status_text)
        self.fingerprint_screen_status_var.set(status_text)
        self.root.update()

    def _make_task_status_callback(self):
        return lambda message: self.root.after(0, lambda: self._update_task_status(message))

    def _run_background_task(self, status_text, worker, on_complete, fingerprint_view=None):
        self._update_task_status(status_text)
        if fingerprint_view is not None:
            self._show_fingerprint_panel(
                title=fingerprint_view["title"],
                prompt=fingerprint_view["prompt"],
                status_text=status_text,
            )
        else:
            self.root.update()

        def runner():
            result = None
            error = None
            try:
                result = worker()
            except Exception as exc:
                error = exc

            def finalize():
                if fingerprint_view is not None:
                    self._hide_fingerprint_panel()
                self._update_auth_mode_status()
                on_complete(result, error)

            self.root.after(0, finalize)

        threading.Thread(target=runner, daemon=True).start()

    def _cancel_door_auto_return(self):
        if self.door_auto_return_after_id is not None:
            try:
                self.root.after_cancel(self.door_auto_return_after_id)
            except Exception:
                pass
            self.door_auto_return_after_id = None
        self.door_countdown_remaining = 0

    def _set_waiting_for_door_open(self, locker_id):
        self.door_title_var.set(f"Dang mo tu so {locker_id}")
        self.door_status_var.set("Dang cho tin hieu mo tu...")

    def _set_waiting_for_door_close(self, locker_id):
        self.door_title_var.set(f"Da mo tu so {locker_id}")
        self.door_status_var.set(f"Da mo tu so {locker_id}.\nDang cho tin hieu dong cua...")

    def _handle_door_open_detected(self, locker_id):
        if self.active_door_locker_id != locker_id or not self.door_frame.winfo_ismapped():
            return
        self._set_waiting_for_door_close(locker_id)

    def _show_door_open_screen(self, locker_id):
        self._cancel_door_auto_return()
        self.active_door_locker_id = locker_id
        self._set_waiting_for_door_open(locker_id)
        self.door_button_frame.pack_forget()
        self.main_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.fingerprint_frame.pack_forget()
        self._hide_admin_panel()
        if not self.door_frame.winfo_ismapped():
            self.door_frame.pack(fill="both", expand=True)
        self.root.update()

    def _start_door_close_monitor(self, locker_id):
        def worker():
            error = None
            opened_seen = True
            self.root.after(0, lambda: self._handle_door_open_detected(locker_id))

            try:
                self.relay_controller.wait_for_door_closed(
                    locker_id,
                    wait_close_seconds=self.DOOR_SENSOR_WAIT_CLOSE_SECONDS,
                )
            except Exception as exc:
                error = exc

            self.root.after(
                0,
                lambda: self._handle_door_close_monitor_done(locker_id, error, opened_seen),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _handle_door_close_monitor_done(self, locker_id, error, opened_seen=False):
        if self.active_door_locker_id != locker_id or not self.door_frame.winfo_ismapped():
            return

        if error is not None:
            if opened_seen:
                self._set_waiting_for_door_close(locker_id)
                self.door_status_var.set(
                    f"Da mo tu so {locker_id}.\nChua nhan duoc tin hieu dong cua: {error}"
                )
            else:
                self._set_waiting_for_door_open(locker_id)
                self.door_status_var.set(
                    f"Dang cho tin hieu mo tu...\n{error}"
                )
            self.door_button_frame.pack(fill="x", pady=(12, 0))
            return

        if not opened_seen:
            self._set_waiting_for_door_open(locker_id)
            self.door_status_var.set(f"Dang cho tin hieu mo tu...\nTu {locker_id} chua xac nhan da mo.")
            self.door_button_frame.pack(fill="x", pady=(12, 0))
            return

        self._set_waiting_for_door_close(locker_id)
        self.door_button_frame.pack(fill="x", pady=(12, 0))
        self._start_door_return_countdown(locker_id)

    def _start_door_return_countdown(self, locker_id):
        self._cancel_door_auto_return()
        self.door_countdown_remaining = self.DOOR_SCREEN_AUTO_RETURN_SECONDS
        self._tick_door_return_countdown(locker_id)

    def _tick_door_return_countdown(self, locker_id):
        if self.active_door_locker_id != locker_id or not self.door_frame.winfo_ismapped():
            self._cancel_door_auto_return()
            return

        if self.door_countdown_remaining <= 0:
            self._return_from_door_screen()
            return

        self.door_status_var.set(
            f"Cua tu so {locker_id} da dong.\n"
            f"Tu dong tro lai sau {self.door_countdown_remaining} giay."
        )
        self.door_countdown_remaining -= 1
        self.door_auto_return_after_id = self.root.after(
            1000,
            lambda: self._tick_door_return_countdown(locker_id),
        )

    def _reopen_active_locker_from_door_screen(self):
        locker_id = self.active_door_locker_id
        if locker_id is None:
            return

        self._cancel_door_auto_return()
        self.door_button_frame.pack_forget()
        self._set_waiting_for_door_open(locker_id)
        if self._is_pc_only_mode():
            self.door_status_var.set(f"Mode 3 dang gia lap mo lai tu so {locker_id} tren may tinh.")
            self.root.after(250, lambda: self._start_door_return_countdown(locker_id))
            return

        def handle_complete(error):
            def finalize():
                if error is not None:
                    self.door_status_var.set(f"Khong mo lai duoc tu so {locker_id}: {error}")
                    self.door_button_frame.pack(fill="x", pady=(12, 0))
                    return

                self._start_door_close_monitor(locker_id)

            self.root.after(0, finalize)

        self.relay_controller.open_locker(locker_id, on_complete=handle_complete)

    def _return_from_door_screen(self):
        self._cancel_door_auto_return()
        self.active_door_locker_id = None
        self.door_frame.pack_forget()
        self.fingerprint_frame.pack_forget()
        self.camera_frame.pack_forget()
        self._hide_admin_panel()
        if not self.main_frame.winfo_ismapped():
            self.main_frame.pack(fill="both", expand=True)
        self.refresh_availability_label()
        self.root.update()

    def _open_locker_async(self, locker_id, on_success, on_error):
        if self._is_pc_only_mode():
            def simulate_success():
                try:
                    on_success()
                except Exception as exc:
                    on_error(exc)

            self.root.after(250, simulate_success)
            return

        if self.relay_controller.has_door_sensor(locker_id):
            self._show_door_open_screen(locker_id)

        def handle_complete(error):
            def finalize():
                if error is not None:
                    if self.active_door_locker_id == locker_id and self.door_frame.winfo_ismapped():
                        self.door_status_var.set(f"Khong mo duoc tu so {locker_id}: {error}")
                        self.door_button_frame.pack(fill="x", pady=(12, 0))
                    self._handle_locker_open_complete(error, on_success, on_error)
                    return

                try:
                    on_success()
                except Exception as exc:
                    on_error(exc)
                    return

                if self.relay_controller.has_door_sensor(locker_id):
                    self._start_door_close_monitor(locker_id)

            self.root.after(0, finalize)

        self.relay_controller.open_locker(locker_id, on_complete=handle_complete)

    @staticmethod
    def _handle_locker_open_complete(error, on_success, on_error):
        if error is not None:
            on_error(error)
            return
        on_success()

    def _recommend_locker_for_item(self, item_profile):
        available_ids = [locker_id for locker_id in self.LOCKER_IDS if locker_id not in self._get_used_locker_ids()]
        if not available_ids:
            return None

        for locker_id in item_profile.get("recommended_lockers", ()):
            if locker_id in available_ids:
                return locker_id
        return available_ids[0]

    def _start_capture_session(self, locker_id, capture_mode="normal"):
        self.pending_mask_capture = capture_mode == "mask"
        self.pending_glasses_capture = capture_mode == "glasses"
        session = FaceCaptureSession(
            user_name=locker_id,
            capture_mode=capture_mode,
            file_tag={
                "normal": "nomask",
                "mask": "mask",
                "glasses": "glasses",
                "mask_glasses": "mask_glasses",
            }.get(capture_mode, "sample"),
        )
        self.start_camera_stream(
            session=session,
            title=f"Dang ky khuon mat cho tu {locker_id}",
            on_complete=lambda captured: self._after_keep_capture(captured, capture_mode),
            cancel_result=0,
            close_before_complete=True,
        )

    def handle_keep_item(self):
        self.pending_item_profile = dict(self.DEFAULT_ITEM_PROFILE)
        recommended_locker = self._recommend_locker_for_item(self.pending_item_profile)
        if recommended_locker is None:
            self.status_var.set("Khong con tu trong de de xuat.")
            messagebox.showwarning(
                "Het tu trong",
                "Hien tai khong con tu trong de gui do.",
            )
            self.pending_item_profile = None
            return

        self.status_var.set(f"San sang dang ky gui do. Du kien su dung tu {recommended_locker}.")
        self.set_busy(True, "Dang nhan dien khuon mat cho thao tac giu do...")
        session = FaceRecognitionSession(
            timeout_seconds=self.RECOGNITION_TIMEOUT_SECONDS,
            min_display_seconds=self.RECOGNITION_TIMEOUT_SECONDS,
            mode='keep', # Che do GUI DO: Chat che, chi mat thuong
        )
        self.start_camera_stream(
            session=session,
            title="Nhan dien de giu do",
            on_complete=self._after_keep_recognition,
            cancel_result={"cancelled": True},
            close_before_complete=False,
        )

    def _after_keep_recognition(self, result):
        continue_keep_flow = False
        try:
            if result.get("cancelled"):
                self._append_event_log("keep_cancelled", result, "Nguoi dung huy thao tac gui do")
                self.status_var.set("Da huy thao tac giu do.")
                return

            if result["name"] != "Unknown":
                self._append_event_log("keep_blocked", result, "Phat hien tu dang duoc su dung")
                self.status_var.set(
                    f"Tu {result['name']} dang duoc su dung ({result['confidence']:.1f}%){self._format_backend_text(result)}."
                )
                messagebox.showinfo(
                    "Thong bao",
                    f"Da nhan dien khuon mat thuoc tu {result['name']}.\n"
                    "Tu nay dang duoc su dung.",
                )
                return

            available_locker_id = self._get_available_locker_id()
            if available_locker_id is None:
                self._append_event_log("keep_rejected", result, "Khong con tu trong")
                self.status_var.set("Ca 3 tu dang duoc su dung. Chi con thao tac lay do.")
                messagebox.showwarning(
                    "Het tu trong",
                    "Ca 3 tu 01, 02, 03 dang duoc su dung.\nKhong the giu do moi, chi co the lay do.",
                )
                return

            item_profile = self.pending_item_profile or {}
            recommended_locker = self._recommend_locker_for_item(item_profile) if item_profile else None
            chosen_locker_id = recommended_locker or available_locker_id
            if chosen_locker_id not in self.LOCKER_IDS or chosen_locker_id in self._get_used_locker_ids():
                chosen_locker_id = available_locker_id

            self.pending_locker_id = chosen_locker_id
            self.pending_used_mask_samples = False
            self.pending_used_glasses_samples = False
            self.status_var.set(f"Dang chup du lieu cho tu {chosen_locker_id}...")
            continue_keep_flow = True
            self._start_capture_session(chosen_locker_id)
        except Exception as exc:
            self.status_var.set("Co loi xay ra trong qua trinh giu do.")
            messagebox.showerror("Loi", str(exc))
        finally:
            if not continue_keep_flow:
                self.pending_item_profile = None
                self.set_busy(False, self.status_var.get())

    def _complete_keep_registration(self, locker_id):
        if self._is_pc_only_mode():
            self._open_keep_locker_after_auth(locker_id, fingerprint_result=None)
            return

        if self._ensure_fingerprint_available():
            self._run_background_task(
                status_text=f"Dat ngon tay len cam bien de dang ky cho tu {locker_id}...",
                worker=lambda: self.fingerprint_controller.enroll_locker(
                    locker_id,
                    progress_callback=self._make_task_status_callback(),
                ),
                fingerprint_view={
                    "title": f"Dang ky van tay cho tu {locker_id}",
                    "prompt": (
                        "Dat ngon tay len cam bien. Khi he thong bao, hay nhac ngon tay ra "
                        "va dat lai lan hai de luu mau van tay."
                    ),
                },
                on_complete=lambda result, error: self._after_keep_fingerprint_enrollment(
                    locker_id,
                    result,
                    error,
                ),
            )
            return

        if self._is_dual_auth_mode():
            self._rollback_keep_registration(
                locker_id,
                (
                    "Mode 1 can ca khuon mat va van tay, nhung cam bien van tay chua san sang.\n"
                    f"{self._fingerprint_unavailable_reason()}"
                ),
            )
            return

        self._open_keep_locker_after_auth(locker_id, fingerprint_result=None)

    def _after_keep_fingerprint_enrollment(self, locker_id, result, error):
        if error is not None:
            if self._is_dual_auth_mode():
                self._rollback_keep_registration(locker_id, str(error))
                return

            messagebox.showwarning(
                "Canh bao",
                f"Khong dang ky duoc van tay cho tu {locker_id}.\n{error}\n"
                "He thong se tiep tuc voi xac thuc khuon mat.",
            )
            self._open_keep_locker_after_auth(locker_id, fingerprint_result=None)
            return

        self._open_keep_locker_after_auth(locker_id, fingerprint_result=result)

    def _open_keep_locker_after_auth(self, locker_id, fingerprint_result=None):
        if self._is_pc_only_mode():
            self.status_var.set(f"Mode 3 dang gia lap mo tu so {locker_id} tren may tinh...")
        else:
            self.status_var.set(f"Dang mo tu so {locker_id} qua relay...")
        self._open_locker_async(
            locker_id=locker_id,
            on_success=lambda: self._finish_keep_after_unlock(locker_id, fingerprint_result),
            on_error=lambda error: self._handle_keep_unlock_error(locker_id, error),
        )

    def _finish_keep_after_unlock(self, locker_id, fingerprint_result=None):
        item_label = (self.pending_item_profile or {}).get("label", "vat dung")
        self.status_var.set(f"Da mo tu so {locker_id}. Dang cap nhat model nen...")
        self._train_user_in_background(locker_id)
        backend_text = "Capture + Relay + Train"
        if self._is_pc_only_mode():
            backend_text = "Capture + PC Mode + Train"
        if fingerprint_result is not None and fingerprint_result.get("matched"):
            backend_text = "Capture + Fingerprint + Relay + Train"
        note = f"Dang ky khuon mat va mo tu thanh cong cho {item_label}"
        self._append_event_log(
            "register_success",
            {"name": locker_id, "confidence": 100.0, "backend": backend_text},
            note,
        )
        self.pending_locker_id = None
        self.pending_item_profile = None
        self.pending_mask_capture = False
        self.pending_glasses_capture = False
        self.pending_used_mask_samples = False
        self.pending_used_glasses_samples = False
        self.refresh_availability_label()
        self.set_training_wait(
            True,
            f"Xin vui long cho doi. Dang cap nhat model cho tu so {locker_id}...",
        )
        self.set_busy(False, self.status_var.get())

    def _rollback_keep_registration(self, locker_id, reason):
        try:
            delete_user(locker_id)
        except Exception:
            pass
        try:
            if not self._is_pc_only_mode():
                self.fingerprint_controller.delete_locker(locker_id)
        except Exception:
            pass
        self._append_event_log(
            "register_failed",
            {"name": locker_id, "confidence": 0.0, "backend": "Face + Fingerprint"},
            reason,
        )
        self.pending_locker_id = None
        self.pending_item_profile = None
        self.pending_mask_capture = False
        self.pending_glasses_capture = False
        self.pending_used_mask_samples = False
        self.pending_used_glasses_samples = False
        self.refresh_availability_label()
        self.set_busy(False, f"Dang ky that bai cho tu {locker_id}.")
        messagebox.showerror("Loi xac thuc", reason)

    def _handle_keep_unlock_error(self, locker_id, error):
        try:
            delete_user(locker_id)
        except Exception:
            pass
        try:
            if not self._is_pc_only_mode():
                self.fingerprint_controller.delete_locker(locker_id)
        except Exception:
            pass
        backend_text = "PC Mode" if self._is_pc_only_mode() else "Relay"
        note = (
            "Gia lap mo tu that bai sau khi chup du lieu"
            if self._is_pc_only_mode()
            else "Mo tu qua relay that bai sau khi chup du lieu"
        )
        self._append_event_log(
            "register_failed",
            {"name": locker_id, "confidence": 0.0, "backend": backend_text},
            note,
        )
        self.pending_locker_id = None
        self.pending_item_profile = None
        self.pending_mask_capture = False
        self.pending_glasses_capture = False
        self.pending_used_mask_samples = False
        self.pending_used_glasses_samples = False
        self.refresh_availability_label()
        self.set_busy(False, f"Khong mo duoc tu so {locker_id}.")
        if self._is_pc_only_mode():
            messagebox.showerror("Loi Mode 3", f"Khong hoan tat gia lap mo tu so {locker_id}.\n{error}")
        else:
            messagebox.showerror("Loi relay", f"Khong mo duoc tu so {locker_id} qua relay.\n{error}")

    def _find_duplicate_registered_locker(self, locker_id):
        try:
            new_vectors = extract_original_face_vectors(locker_id)
        except Exception:
            return None

        if not new_vectors:
            return None

        existing_samples = []
        for known_locker_id in self.LOCKER_IDS:
            if known_locker_id == locker_id:
                continue
            try:
                known_vectors = extract_original_face_vectors(known_locker_id)
            except Exception:
                known_vectors = []
            for known_vector in known_vectors:
                existing_samples.append((known_vector, known_locker_id))

        if not existing_samples:
            return None

        match_counts = Counter()
        best_distances = {}
        comparable_count = 0

        for face_vector in new_vectors:
            best_name = None
            best_distance = None
            for known_vector, known_name in existing_samples:
                distance = float(np.linalg.norm(face_vector - known_vector))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_name = known_name

            comparable_count += 1
            if (
                best_name is not None
                and best_distance is not None
                and best_distance <= self.DUPLICATE_FACE_DISTANCE_THRESHOLD
            ):
                match_counts[best_name] += 1
                previous_best = best_distances.get(best_name)
                best_distances[best_name] = (
                    best_distance if previous_best is None else min(previous_best, best_distance)
                )

        if not match_counts or comparable_count <= 0:
            return None

        required_matches = max(
            self.DUPLICATE_FACE_MIN_MATCHES,
            int(np.ceil(comparable_count * self.DUPLICATE_FACE_MATCH_RATIO)),
        )
        matched_locker, match_count = match_counts.most_common(1)[0]
        if match_count < required_matches:
            return None

        return {
            "locker_id": matched_locker,
            "match_count": match_count,
            "sample_count": comparable_count,
            "best_distance": best_distances.get(matched_locker, 1.0),
        }

    def _cancel_duplicate_registration(self, locker_id, duplicate_match):
        matched_locker = duplicate_match["locker_id"]
        note = (
            f"Phat hien khuon mat trung voi tu {matched_locker} "
            f"({duplicate_match['match_count']}/{duplicate_match['sample_count']} mau)."
        )
        self._append_event_log(
            "register_failed",
            {"name": locker_id, "confidence": 0.0, "backend": "DuplicateCheck"},
            note,
        )
        try:
            delete_user(locker_id)
        except Exception:
            pass
        self.pending_locker_id = None
        self.pending_item_profile = None
        self.pending_mask_capture = False
        self.pending_glasses_capture = False
        self.pending_used_mask_samples = False
        self.pending_used_glasses_samples = False
        self.refresh_availability_label()
        self.set_busy(False, f"Khuon mat nay da duoc dang ky o tu {matched_locker}.")
        messagebox.showwarning(
            "Trung khuon mat",
            (
                f"Khuon mat nay da duoc dang ky o tu so {matched_locker}.\n"
                "Moi nguoi chi duoc dang ky 1 tu, nen he thong da huy dang ky moi."
            ),
        )

    def _find_face_box_for_validation(self, image, image_path):
        metadata_path = os.path.splitext(image_path)[0] + ".json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                face_box = metadata.get("face_box")
                if isinstance(face_box, list) and len(face_box) == 4:
                    return tuple(int(value) for value in face_box)
            except Exception:
                pass

        small_frame = cv2.resize(
            image,
            (0, 0),
            fx=FRAME_PROCESS_SCALE,
            fy=FRAME_PROCESS_SCALE,
        )
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        boxes, _backend = detect_face_boxes(rgb_small)
        if len(boxes) != 1:
            return None

        scale_back = 1.0 / FRAME_PROCESS_SCALE
        return tuple(int(round(value * scale_back)) for value in boxes[0])

    def _validate_normal_capture_samples(self, locker_id):
        user_dir = os.path.join("dataset", str(locker_id))
        if not os.path.isdir(user_dir):
            return False, "Khong tim thay thu muc mau vua chup."

        image_paths = []
        try:
            for file_name in os.listdir(user_dir):
                lower_name = file_name.lower()
                if (
                    lower_name.endswith((".jpg", ".jpeg", ".png"))
                    and "_nomask_" in lower_name
                    and "_synthetic_" not in lower_name
                ):
                    image_paths.append(os.path.join(user_dir, file_name))
        except OSError as exc:
            return False, str(exc)

        if not image_paths:
            return False, "Khong co mau mat tran de kiem tra."

        state_model = load_state_svm()
        invalid_samples = []
        for image_path in sorted(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                invalid_samples.append((os.path.basename(image_path), "khong doc duoc anh"))
                continue

            face_box = self._find_face_box_for_validation(image, image_path)
            if face_box is None:
                invalid_samples.append((os.path.basename(image_path), "khong co dung 1 khuon mat"))
                continue

            face_state = classify_face_state(image, face_box, state_model)
            is_frontal, _pose_metrics = is_face_frontal_enough(image, face_box)
            if face_state != "normal":
                invalid_samples.append((os.path.basename(image_path), face_state))
            elif not is_frontal:
                invalid_samples.append((os.path.basename(image_path), "mat nghieng"))

        if invalid_samples:
            sample_name, reason = invalid_samples[0]
            return (
                False,
                (
                    "Bo mau vua chup khong phai mat tran on dinh "
                    f"({sample_name}: {reason}). Vui long bo kinh/khau trang va thu lai."
                ),
            )

        return True, ""

    def _after_keep_capture(self, captured, capture_mode):
        try:
            self._emit_mode3_result(
                "keep_capture",
                {"name": self.pending_locker_id or "-", "confidence": 100.0, "backend": "Capture"},
                extra_text=f"captured={captured} mode={capture_mode}",
            )
            if captured <= 0:
                self._discard_pending_capture_data(self.pending_locker_id)
                self._append_event_log(
                    "register_failed",
                    {"name": self.pending_locker_id or "", "confidence": 0.0, "backend": "Capture"},
                    "Khong chup duoc mau hop le",
                )
                self.status_var.set("Khong chup duoc anh nao. Huy thao tac.")
                messagebox.showwarning("Canh bao", "Khong chup duoc anh nao de dang ky.")
                self.pending_locker_id = None
                self.pending_item_profile = None
                self.pending_mask_capture = False
                self.pending_glasses_capture = False
                self.pending_used_mask_samples = False
                self.pending_used_glasses_samples = False
                self.refresh_availability_label()
                self.set_busy(False, self.status_var.get())
                return

            locker_id = self.pending_locker_id
            if locker_id is None:
                raise RuntimeError("Khong tim thay tu dang ky de tiep tuc.")

            if capture_mode == "normal":
                samples_ok, validation_error = self._validate_normal_capture_samples(locker_id)
                if not samples_ok:
                    self._discard_pending_capture_data(locker_id)
                    self._append_event_log(
                        "register_failed",
                        {"name": locker_id, "confidence": 0.0, "backend": "NormalSampleValidation"},
                        validation_error,
                    )
                    self.pending_locker_id = None
                    self.pending_item_profile = None
                    self.pending_mask_capture = False
                    self.pending_glasses_capture = False
                    self.pending_used_mask_samples = False
                    self.pending_used_glasses_samples = False
                    self.refresh_availability_label()
                    self.status_var.set("Da huy bo mau vi phat hien phu kien hoac mat khong on dinh.")
                    self.set_busy(False, self.status_var.get())
                    messagebox.showwarning("Mau khong hop le", validation_error)
                    return

                duplicate_match = self._find_duplicate_registered_locker(locker_id)
                if duplicate_match is not None:
                    self._cancel_duplicate_registration(locker_id, duplicate_match)
                    return

                self._complete_keep_registration(locker_id)
                return
        except Exception as exc:
            self._discard_pending_capture_data(self.pending_locker_id)
            self.pending_locker_id = None
            self.pending_item_profile = None
            self.pending_mask_capture = False
            self.pending_glasses_capture = False
            self.pending_used_mask_samples = False
            self.pending_used_glasses_samples = False
            self.refresh_availability_label()
            self.status_var.set("Co loi xay ra trong qua trinh giu do.")
            self.set_busy(False, self.status_var.get())
            messagebox.showerror("Loi", str(exc))

    def show_take_options(self):
        options_window = tk.Toplevel(self.root)
        options_window.title("Chon cach lay do")
        option_width_px = 420
        option_height_px = 400
        options_window.resizable(False, False)
        self._place_child_window(options_window, option_width_px, option_height_px)
        options_window.configure(bg=self.BG_COLOR)

        info_label = tk.Label(
            options_window,
            text="Chon hinh thuc lay do:",
            font=("Segoe UI", 18, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.BG_COLOR,
            pady=20,
        )
        info_label.pack()

        description_label = tk.Label(
            options_window,
            text="Lay tam thoi se giu nguyen du lieu khuon mat. Ket thuc gui se xoa du lieu da dang ky.",
            font=("Segoe UI", 12),
            fg=self.MUTED_COLOR,
            bg=self.BG_COLOR,
            wraplength=min(380, self.content_wrap_px),
            justify="center",
        )
        description_label.pack()

        button_frame = tk.Frame(options_window, bg=self.BG_COLOR, pady=20)
        button_frame.pack(fill="x", padx=25)
        button_frame.grid_columnconfigure(0, weight=1)

        temporary_button = tk.Button(
            button_frame,
            text="Lay do tam thoi",
            font=("Segoe UI", 16, "bold"),
            bg=self.PRIMARY_COLOR,
            fg="white",
            activebackground=self.PRIMARY_ACTIVE_BG,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=10,
            pady=16,
            cursor="hand2",
            command=lambda: self._handle_take_option(options_window, temporary=True),
        )
        temporary_button.grid(row=0, column=0, sticky="ew")

        finish_button = tk.Button(
            button_frame,
            text="Ket thuc gui",
            font=("Segoe UI", 16, "bold"),
            bg=self.SECONDARY_COLOR,
            fg="white",
            activebackground=self.SECONDARY_ACTIVE_BG,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=10,
            pady=16,
            cursor="hand2",
            command=lambda: self._handle_take_option(options_window, temporary=False),
        )
        finish_button.grid(row=1, column=0, pady=(15, 0), sticky="ew")

        cancel_button = tk.Button(
            options_window,
            text="Dong",
            width=12,
            font=("Segoe UI", 12, "bold"),
            bg=self.SOFT_BUTTON_BG,
            fg=self.TEXT_COLOR,
            activebackground=self.SOFT_BUTTON_ACTIVE_BG,
            relief="flat",
            bd=0,
            cursor="hand2",
            command=options_window.destroy,
        )
        cancel_button.pack(pady=10)

    def _handle_take_option(self, options_window, temporary):
        options_window.destroy()
        self.handle_take_item(temporary=temporary)

    def handle_take_item(self, temporary=False):
        self._start_take_item_recognition(temporary=temporary, verification_pass=1, expected_name=None)

    def _start_take_item_recognition(self, temporary=False, verification_pass=1, expected_name=None):
        action_text = "lay do tam thoi" if temporary else "lay do ket thuc gui"
        if verification_pass > 1 and expected_name:
            self.set_busy(
                True,
                (
                    f"Dang xac minh them cho tu {expected_name} "
                    f"({self.RECOGNITION_TIMEOUT_SECONDS + self.TAKE_REVIEW_TIMEOUT_SECONDS} giay)..."
                ),
            )
        else:
            self.set_busy(True, f"Dang nhan dien khuon mat cho thao tac {action_text}...")

        timeout_seconds = self.RECOGNITION_TIMEOUT_SECONDS
        if verification_pass > 1:
            timeout_seconds += self.TAKE_REVIEW_TIMEOUT_SECONDS

        session = FaceRecognitionSession(
            timeout_seconds=timeout_seconds,
            min_display_seconds=timeout_seconds,
            mode='take', # Che do LAY DO: Linh hoat, chap nhan phu kien
        )
        self.start_camera_stream(
            session=session,
            title="Nhan dien khi lay do" if verification_pass == 1 else "Xac minh them khi lay do",
            on_complete=lambda result: self._after_take_recognition(
                result,
                temporary,
                verification_pass=verification_pass,
                expected_name=expected_name,
            ),
            cancel_result={"cancelled": True},
            close_before_complete=True,
        )

    def _after_take_recognition(self, result, temporary, verification_pass=1, expected_name=None):
        keep_busy = False
        try:
            self._emit_mode3_result(
                "take_recognition_received",
                result,
                extra_text=(
                    f"temporary={temporary} pass={verification_pass}"
                    + (f" expected={expected_name}" if expected_name else "")
                ),
            )
            if result.get("cancelled"):
                self._append_event_log("take_cancelled", result, "Nguoi dung huy thao tac lay do")
                self.status_var.set("Da huy thao tac lay do.")
                return

            if result["name"] == "Unknown":
                if (
                    not self._is_pc_only_mode()
                    and not self._is_dual_auth_mode()
                    and self._ensure_fingerprint_available()
                ):
                    keep_busy = True
                    self._run_background_task(
                        status_text="Dat ngon tay len cam bien de xac thuc mo tu...",
                        worker=lambda: self.fingerprint_controller.verify_locker(),
                        fingerprint_view={
                            "title": "Xac thuc van tay",
                            "prompt": (
                                "Khong nhan dien duoc khuon mat. Dat ngon tay da dang ky "
                                "len cam bien de xac dinh tu can mo."
                            ),
                        },
                        on_complete=lambda fp_result, error: self._after_take_fingerprint_check(
                            face_result=result,
                            fingerprint_result=fp_result,
                            error=error,
                            temporary=temporary,
                            strict_mode=False,
                        ),
                    )
                else:
                    self._reject_take_result(
                        result,
                        "Khong xac thuc duoc nguoi lay do bang khuon mat.",
                    )
                return

            required_confidence = self._get_take_confidence_threshold(result)
            review_confidence = self._get_take_review_confidence_threshold(result)
            recheck_accept_confidence = self._get_take_recheck_accept_confidence_threshold(result)
            actual_confidence = float(result.get("confidence", 0.0))
            predicted_name = result["name"]

            if verification_pass > 1:
                if predicted_name == expected_name and actual_confidence >= review_confidence:
                    result["backend"] = f"{result.get('backend', '')} + Recheck"
                    if actual_confidence < required_confidence:
                        if actual_confidence >= recheck_accept_confidence:
                            result["backend"] = f"{result.get('backend', '')} + BorderlinePass"
                        else:
                            self._append_event_log(
                                "take_rejected",
                                result,
                                (
                                    "Xac minh bo sung ra dung tu nhung do tin cay van chua du "
                                    f"({actual_confidence:.1f}% < {recheck_accept_confidence:.1f}%)."
                                ),
                            )
                            self.status_var.set(
                                "Da xac minh them nhung do tin cay van chua du de mo tu"
                                f"{self._format_backend_text(result)}."
                            )
                            messagebox.showwarning(
                                "Tu choi",
                                (
                                    f"He thong da xac minh them va van ra tu {predicted_name}, "
                                    f"nhung do tin cay moi dat {actual_confidence:.1f}%.\n"
                                    f"Can it nhat {recheck_accept_confidence:.1f}% sau vong xac minh them.\n"
                                    "Vui long dieu chinh goc mat, anh sang va thu lai."
                                ),
                            )
                            return
                else:
                    self._append_event_log(
                        "take_rejected",
                        result,
                        (
                            "Xac minh bo sung that bai "
                            f"(du doan={predicted_name}, mong doi={expected_name}, "
                            f"tin cay={actual_confidence:.1f}%)."
                        ),
                    )
                    self.status_var.set(
                        "Da xac minh them nhung ket qua chua du on dinh"
                        f"{self._format_backend_text(result)}."
                    )
                    messagebox.showwarning(
                        "Tu choi",
                        (
                            "He thong da thu xac minh them nhung ket qua van chua du chac chan.\n"
                            f"Lan cuoi du doan: {predicted_name} ({actual_confidence:.1f}%).\n"
                            "Vui long dieu chinh goc mat, anh sang va thu lai."
                        ),
                    )
                    return

            if actual_confidence < required_confidence:
                if verification_pass == 1 and actual_confidence >= review_confidence:
                    keep_busy = True
                    self._emit_mode3_result(
                        "take_recognition_recheck",
                        result,
                        extra_text=f"confidence={actual_confidence:.1f} threshold={required_confidence:.1f}",
                    )
                    self.status_var.set(
                        (
                            f"Do tin cay dang o vung xac minh them ({actual_confidence:.1f}%). "
                            "Vui long giu yen them mot chut..."
                        )
                    )
                    self._start_take_item_recognition(
                        temporary=temporary,
                        verification_pass=2,
                        expected_name=predicted_name,
                    )
                    return

                self._append_event_log(
                    "take_rejected",
                    result,
                    (
                        "Do tin cay nhan dien qua thap de mo tu "
                        f"({actual_confidence:.1f}% < {required_confidence:.1f}%)."
                    ),
                )
                self.status_var.set(
                    f"Nhan dien chua du chac chan ({actual_confidence:.1f}% < {required_confidence:.1f}%)"
                    f"{self._format_backend_text(result)}."
                )
                messagebox.showwarning(
                    "Tu choi",
                    (
                        f"He thong nhan dien ra tu {result['name']} nhung do tin cay chi dat "
                        f"{actual_confidence:.1f}%.\n"
                        f"Can toi thieu {required_confidence:.1f}% moi duoc mo tu.\n"
                        "Vui long dung thang camera, giu anh sang tot va thu lai."
                    ),
                )
                return

            if self._is_dual_auth_mode():
                if not self._ensure_fingerprint_available():
                    self._reject_take_result(
                        result,
                        (
                            "Mode 1 yeu cau cam bien van tay, nhung cam bien chua san sang.\n"
                            f"{self._fingerprint_unavailable_reason()}"
                        ),
                    )
                    return

                keep_busy = True
                locker_id = result["name"]
                self._run_background_task(
                    status_text=f"Dat ngon tay len cam bien de xac thuc tu {locker_id}...",
                    worker=lambda: self.fingerprint_controller.verify_locker(expected_locker_id=locker_id),
                    fingerprint_view={
                        "title": f"Xac thuc van tay tu {locker_id}",
                        "prompt": (
                            "Khuon mat da hop le. Dat dung ngon tay da dang ky cua tu nay "
                            "len cam bien de mo tu."
                        ),
                    },
                    on_complete=lambda fp_result, error: self._after_take_fingerprint_check(
                        face_result=result,
                        fingerprint_result=fp_result,
                        error=error,
                        temporary=temporary,
                        strict_mode=True,
                    ),
                )
                return

            keep_busy = True
            self._open_take_locker_after_auth(result["name"], temporary, result)
        except Exception as exc:
            self.status_var.set("Co loi xay ra trong qua trinh lay do.")
            messagebox.showerror("Loi", str(exc))
        finally:
            self.refresh_availability_label()
            if not keep_busy:
                self.set_busy(False, self.status_var.get())

    def _after_take_fingerprint_check(
        self,
        face_result,
        fingerprint_result,
        error,
        temporary,
        strict_mode,
    ):
        if error is not None:
            self._reject_take_result(face_result, str(error))
            return

        if not fingerprint_result or not fingerprint_result.get("matched"):
            reason = (
                fingerprint_result.get("error")
                if fingerprint_result is not None and fingerprint_result.get("error")
                else "Khong xac thuc duoc van tay hop le."
            )
            self._reject_take_result(face_result, reason)
            return

        locker_id = fingerprint_result["locker_id"]
        if strict_mode and face_result["name"] != locker_id:
            self._reject_take_result(
                face_result,
                f"Van tay thuoc tu {locker_id}, khong trung voi khuon mat {face_result['name']}.",
            )
            return

        if strict_mode:
            result = dict(face_result)
            result["backend"] = f"{face_result.get('backend', 'Face')} + Fingerprint"
            result["confidence"] = min(
                float(face_result.get("confidence", 0.0)),
                float(fingerprint_result.get("confidence", 100.0)),
            )
        else:
            result = {
                "name": locker_id,
                "confidence": float(fingerprint_result.get("confidence", 100.0)),
                "backend": f"Fingerprint ({fingerprint_result.get('backend', 'Unknown')})",
            }

        self._open_take_locker_after_auth(locker_id, temporary, result)

    def _open_take_locker_after_auth(self, locker_id, temporary, result):
        if self._is_pc_only_mode():
            result = dict(result)
            result["backend"] = f"{result.get('backend', 'Face')} + PC Mode"
            self.status_var.set(f"Mode 3 dang gia lap mo tu so {locker_id} tren may tinh...")
            self._emit_mode3_result(
                "take_unlock_started",
                result,
                extra_text=f"locker={locker_id} temporary={temporary}",
            )
        else:
            self.status_var.set(f"Dang mo tu so {locker_id} qua relay...")
        self._open_locker_async(
            locker_id=locker_id,
            on_success=lambda: self._finish_take_after_unlock(locker_id, temporary, result),
            on_error=lambda error: self._handle_take_unlock_error(locker_id, error),
        )

    def _finish_take_after_unlock(self, locker_id, temporary, result):
        self._emit_mode3_result(
            "take_unlock_success",
            result,
            extra_text=f"locker={locker_id} temporary={temporary}",
        )
        if temporary:
            self._append_event_log("take_temporary", result, "Lay do tam thoi thanh cong")
            self.status_var.set(
                f"Da mo tu so {locker_id} tam thoi{self._format_backend_text(result)}."
            )
        else:
            delete_user(locker_id)
            try:
                if not self._is_pc_only_mode():
                    self.fingerprint_controller.delete_locker(locker_id)
            except Exception:
                pass
            self._append_event_log("take_finish", result, "Ket thuc gui va xoa du lieu khuon mat")
            self.status_var.set(
                f"Da mo tu so {locker_id} va xoa du lieu{self._format_backend_text(result)}."
            )
        self.refresh_availability_label()
        self.set_busy(False, self.status_var.get())

    def _handle_take_unlock_error(self, locker_id, error):
        self.refresh_availability_label()
        self.set_busy(False, f"Khong mo duoc tu so {locker_id}.")
        if self._is_pc_only_mode():
            messagebox.showerror("Loi Mode 3", f"Khong hoan tat gia lap mo tu so {locker_id}.\n{error}")
        else:
            messagebox.showerror("Loi relay", f"Khong mo duoc tu so {locker_id} qua relay.\n{error}")

    def _reject_take_result(self, result, reason):
        self._emit_mode3_result("take_rejected", result, extra_text=reason)
        self._append_event_log("take_rejected", result, reason)
        self.status_var.set(reason)
        self.refresh_availability_label()
        self.set_busy(False, self.status_var.get())
        messagebox.showwarning("Tu choi", reason)

    def start_camera_stream(self, session, title, on_complete, cancel_result, close_before_complete=False):
        self.pending_event_snapshot_frame = None
        self.camera_session = session
        self.camera_callback = on_complete
        self.camera_cancel_result = cancel_result
        self.camera_close_before_complete = close_before_complete
        self.mode3_terminal_last_packet = ""
        self.mode3_terminal_last_result = ""
        self.camera_title_var.set(title)
        self.camera_status_var.set("Dang khoi dong camera...")
        self._print_mode3_terminal(f"start_camera_stream: title={title}")

        if not self.camera_frame.winfo_ismapped():
            self.main_frame.pack_forget()
            self.fingerprint_frame.pack_forget()
            self.door_frame.pack_forget()
            self._hide_admin_panel()
            self.camera_frame.pack(fill="both", expand=True)
            self.root.update_idletasks()

        if self.camera_cap is None:
            self.camera_cap = self._open_camera(None)
            if not self.camera_cap.isOpened():
                self.stop_camera_stream()
                raise RuntimeError("Cannot open webcam.")

            self.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
            self.camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self.camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.camera_after_id is None:
            self.camera_after_id = self.root.after(0, self._camera_loop)

    def _camera_loop(self):
        if self.camera_cap is None or self.camera_session is None:
            return
        self.camera_after_id = None

        ret, frame = self.camera_cap.read()
        if not ret:
            self.stop_camera_stream()
            messagebox.showerror("Loi", "Khong doc duoc khung hinh tu webcam.")
            self.set_busy(False, "Co loi xay ra khi doc webcam.")
            return

        packet = self.camera_session.process_frame(frame)
        self.latest_camera_frame = packet["frame"].copy()
        self._emit_mode3_camera_packet(packet)
        self.camera_status_var.set(
            (
                f"{packet.get('status_text', '')}"
                f" | FPS: {int(packet.get('fps', 0))}"
            )
            if "fps" in packet
            else f"{packet.get('status_text', '')}"
        )
        self._render_camera_frame(packet["frame"])

        if packet.get("done"):
            result = packet.get("result")
            self._emit_mode3_result("camera_done", result)
            callback = self.camera_callback
            close_before_complete = self.camera_close_before_complete
            self.camera_session = None
            self.camera_callback = None
            self.camera_cancel_result = None
            if callback is not None:
                self.root.after(
                    0,
                    lambda: self._handle_camera_completion(
                        callback,
                        result,
                        close_before_complete=close_before_complete,
                    ),
                )
            else:
                self.stop_camera_stream()
            return

        self.camera_after_id = self.root.after(30, self._camera_loop)

    def cancel_camera_stream(self):
        callback = self.camera_callback
        cancel_result = self.camera_cancel_result
        self.stop_camera_stream()
        if callback is not None:
            self.root.after(0, lambda: callback(cancel_result))

    def _handle_camera_completion(self, callback, result, close_before_complete=False):
        preserved_frame = self.latest_camera_frame.copy() if self.latest_camera_frame is not None else None
        if preserved_frame is not None:
            self.pending_event_snapshot_frame = preserved_frame
        if close_before_complete:
            self.stop_camera_stream()
            self.latest_camera_frame = preserved_frame
        callback(result)
        if self.camera_session is None:
            self.stop_camera_stream()
        elif self.camera_after_id is None:
            self.camera_after_id = self.root.after(0, self._camera_loop)

    def _train_user_in_background(self, locker_id):
        def worker():
            try:
                train_user(user_name=locker_id)
            except Exception as exc:
                self.root.after(0, lambda: self._finish_background_training(locker_id, str(exc)))
                return

            self.root.after(0, lambda: self._finish_background_training(locker_id, None))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_background_training(self, locker_id, error_message):
        self.refresh_availability_label()
        self.set_training_wait(False)

        if error_message:
            if not self.is_action_busy:
                self.status_var.set("Co loi xay ra trong qua trinh giu do.")
                self.root.update()
            messagebox.showerror("Loi", error_message)
            return

        if not self.is_action_busy:
            self.status_var.set(
                f"Da cap nhat model cho tu so {locker_id}. Co the tiep tuc su dung."
            )
            self.root.update()

    def stop_camera_stream(self):
        if self.camera_after_id is not None:
            self.root.after_cancel(self.camera_after_id)
            self.camera_after_id = None

        if self.camera_cap is not None:
            self.camera_cap.release()
            self.camera_cap = None

        self.camera_session = None
        self.camera_callback = None
        self.camera_cancel_result = None
        self.camera_photo = None
        self.latest_camera_frame = None
        self.camera_preview.config(image="", bg=self.CAMERA_BG, width=1, height=1)
        self.camera_frame.pack_forget()
        if (
            not self.fingerprint_frame.winfo_ismapped()
            and not self.door_frame.winfo_ismapped()
            and getattr(self, "admin_panel_frame", None) is None
        ):
            self.main_frame.pack(fill="both", expand=True)

    @staticmethod
    def _open_camera(camera_index):
        return create_camera(
            camera_index=camera_index,
            width=CAMERA_FRAME_WIDTH,
            height=CAMERA_FRAME_HEIGHT,
        )

    def _render_camera_frame(self, frame):
        target_width = max(1, self.camera_preview_wrap.winfo_width())
        target_height = max(1, self.camera_preview_wrap.winfo_height())
        if target_width <= 1 or target_height <= 1:
            target_width = min(self.CAMERA_PANEL_WIDTH, self.window_width_px)
            target_height = min(self.CAMERA_PANEL_HEIGHT, self.window_height_px)

        if self.window_height_px >= self.window_width_px:
            max_preview_width = max(1, self.window_width_px - 20)
            max_preview_height = max(1, self.window_height_px - 190)
            portrait_width = min(target_width, max_preview_width)
            portrait_height = min(target_height, max_preview_height)
            if portrait_height / max(portrait_width, 1) < 1.15:
                portrait_height = min(max_preview_height, int(round(portrait_width / self.CAMERA_PORTRAIT_ASPECT)))
            target_width = max(1, portrait_width)
            target_height = max(1, portrait_height)

        frame_height, frame_width = frame.shape[:2]
        display_frame = frame
        if self.window_height_px >= self.window_width_px:
            target_aspect = target_width / target_height
            frame_aspect = frame_width / frame_height
            if frame_aspect > target_aspect:
                crop_width = max(1, int(round(frame_height * target_aspect)))
                crop_left = max(0, (frame_width - crop_width) // 2)
                display_frame = frame[:, crop_left:crop_left + crop_width]
            elif frame_aspect < target_aspect:
                crop_height = max(1, int(round(frame_width / target_aspect)))
                crop_top = max(0, (frame_height - crop_height) // 2)
                display_frame = frame[crop_top:crop_top + crop_height, :]

        frame_height, frame_width = display_frame.shape[:2]
        if self.CAMERA_DISPLAY_MIRROR:
            display_frame = cv2.flip(display_frame, 1)

        frame_height, frame_width = display_frame.shape[:2]
        scale = min(target_width / frame_width, target_height / frame_height)
        scaled_width = max(1, int(round(frame_width * scale)))
        scaled_height = max(1, int(round(frame_height * scale)))

        resized_frame = cv2.resize(display_frame, (scaled_width, scaled_height))
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas[:, :] = self._hex_to_bgr(self.STATUS_COLOR)
        offset_x = (target_width - scaled_width) // 2
        offset_y = (target_height - scaled_height) // 2
        canvas[offset_y:offset_y + scaled_height, offset_x:offset_x + scaled_width] = resized_frame

        rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image = self._apply_camera_overlay(image)
        self.camera_photo = ImageTk.PhotoImage(image=image)
        self.camera_preview.config(
            image=self.camera_photo,
            width=target_width,
            height=target_height,
        )

    def _apply_camera_overlay(self, image):
        width, height = image.size
        if width < 80 or height < 80:
            return image

        face_width = int(width * 0.72)
        face_height = int(height * 0.74)
        left = (width - face_width) // 2
        top = (height - face_height) // 2
        right = left + face_width
        bottom = top + face_height

        composed = image.convert("RGBA")
        shade = (0, 0, 0, 112)
        overlay = Image.new("RGBA", (width, height), shade)
        overlay_alpha = Image.new("L", (width, height), shade[3])
        oval_mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(oval_mask)
        mask_draw.ellipse((left, top, right, bottom), fill=255)
        overlay_alpha.paste(0, mask=oval_mask)
        overlay.putalpha(overlay_alpha)
        composed = Image.alpha_composite(composed, overlay)

        draw = ImageDraw.Draw(composed, "RGBA")
        line_width = max(3, width // 140)
        frame_color = _hex_to_rgba(self.PRIMARY_COLOR, 245)
        glow_color = _hex_to_rgba("#FFFFFF", 95)
        for color, stroke_width in ((glow_color, line_width + 3), (frame_color, line_width)):
            draw.ellipse((left, top, right, bottom), outline=color, width=stroke_width)

        return composed.convert("RGB")

    @staticmethod
    def _hex_to_bgr(hex_color):
        hex_value = hex_color.lstrip("#")
        if len(hex_value) != 6:
            return (17, 17, 17)
        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)
        return (blue, green, red)

    @staticmethod
    def _hex_to_rgb(hex_color):
        hex_value = hex_color.lstrip("#")
        if len(hex_value) != 6:
            return (17, 17, 17)
        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)
        return (red, green, blue)

    def _load_locker_preview(self, locker_id):
        locker_dir = os.path.join("dataset", locker_id)
        if not os.path.isdir(locker_dir):
            return None

        real_candidates = []
        synthetic_candidates = []
        for file_name in os.listdir(locker_dir):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            file_path = os.path.join(locker_dir, file_name)
            try:
                modified_time = os.path.getmtime(file_path)
            except OSError:
                continue
            if "_synthetic_" in file_name.lower():
                synthetic_candidates.append((modified_time, file_path))
            else:
                real_candidates.append((modified_time, file_path))

        candidates = real_candidates or synthetic_candidates
        if not candidates:
            return None

        candidates.sort(reverse=True)
        image_path = candidates[0][1]

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail(self.ADMIN_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        except Exception:
            return None

        thumb_background = Image.new(
            "RGB",
            self.ADMIN_THUMBNAIL_SIZE,
            self._hex_to_rgb(self.STATUS_COLOR),
        )
        offset_x = (self.ADMIN_THUMBNAIL_SIZE[0] - image.width) // 2
        offset_y = (self.ADMIN_THUMBNAIL_SIZE[1] - image.height) // 2
        thumb_background.paste(image, (offset_x, offset_y))
        return ImageTk.PhotoImage(thumb_background)

    def _get_available_locker_id(self):
        used_ids = self._get_used_locker_ids()
        for locker_id in self.LOCKER_IDS:
            if locker_id not in used_ids:
                return locker_id
        return None

    def _get_used_locker_ids(self):
        used_ids = set()
        try:
            data = load_model()
            used_ids.update(name for name in data.get("names", []) if name in self.LOCKER_IDS)
        except Exception:
            pass

        dataset_dir = "dataset"
        if os.path.isdir(dataset_dir):
            for folder_name in os.listdir(dataset_dir):
                folder_path = os.path.join(dataset_dir, folder_name)
                if os.path.isdir(folder_path) and folder_name in self.LOCKER_IDS:
                    used_ids.add(folder_name)

        return used_ids

    def refresh_availability_label(self):
        used_ids = self._get_used_locker_ids()
        used_count = len(used_ids)
        available_count = len(self.LOCKER_IDS) - used_count
        label = f"Con {available_count} tu" if available_count > 0 else "Day tu"
        self.available_var.set(label)
        if available_count > 0:
            availability_note = f"Con {available_count} tu san sang cho thao tac gui do."
        else:
            availability_note = "Tat ca tu dang duoc su dung. Chi con thao tac lay do."

        try:
            health = inspect_model_consistency()
        except Exception:
            health = None
        try:
            separation = inspect_identity_separation(top_k=3)
        except Exception:
            separation = None

        health_warning = ""
        if health is not None:
            issues = []
            if health["missing_in_model"]:
                issues.append(f"dataset chua train: {', '.join(health['missing_in_model'])}")
            if health["missing_in_dataset"]:
                issues.append(f"model du nhan: {', '.join(health['missing_in_dataset'])}")
            if used_count > 0 and not health["has_upper_features"]:
                issues.append("chua co upper-face feature")
            if separation is not None:
                high_risk_pairs = [row for row in separation.get("risk_pairs", []) if row.get("risk_level") == "high"]
                if high_risk_pairs:
                    pair = high_risk_pairs[0]
                    issues.append(
                        "cap de nham: "
                        f"{pair['user_a']}-{pair['user_b']} "
                        f"(d={pair['min_distance']:.3f})"
                    )
            health_warning = " | ".join(issues)

        self.last_health_warning = health_warning
        if health_warning:
            self.available_note_var.set(f"{availability_note} Can kiem tra: {health_warning}.")
        else:
            self.available_note_var.set(availability_note)
        self.availability_badge.config(
            bg=self.DANGER_COLOR if used_count >= len(self.LOCKER_IDS) else self.PRIMARY_COLOR
        )
        self._draw_locker_status(used_ids)

    def show_admin_login(self):
        if self.is_action_busy or self.is_training_model:
            messagebox.showwarning("Ban", "Vui long cho hanh dong hien tai ket thuc.")
            return

        self.main_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.fingerprint_frame.pack_forget()
        self.door_frame.pack_forget()
        self._hide_admin_panel()

        login_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=14, pady=14)
        self.admin_panel_frame = login_frame
        login_frame.pack(fill="both", expand=True)
        login_content = self._make_scrollable_page(login_frame)

        header_frame = tk.Frame(
            login_content,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=16,
            pady=16,
        )
        header_frame.pack(fill="x", padx=20, pady=(20, 0))
        header_frame.grid_columnconfigure(0, weight=1)

        copy_frame = tk.Frame(header_frame, bg=self.PANEL_COLOR)
        copy_frame.grid(row=0, column=0, sticky="w")

        tk.Label(
            copy_frame,
            text="TRUY CAP QUAN TRI",
            font=("Segoe UI Semibold", 9, "bold"),
            fg=self.HEADER_ACCENT,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        tk.Label(
            copy_frame,
            text="Dang nhap quan tri",
            font=("Segoe UI Semibold", 16, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            copy_frame,
            text="Nhap ma PIN 6 so de mo bang dieu khien quan ly tu.",
            font=self.BODY_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(header_frame, image=self.logo_photo, bg=self.PANEL_COLOR).grid(row=0, column=1, padx=(10, 0), sticky="e")

        form_frame = tk.Frame(
            login_content,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=16,
            pady=16,
        )
        form_frame.pack(fill="x", padx=20, pady=(16, 0))

        tk.Label(
            form_frame,
            text="Ma PIN quan tri:",
            font=("Segoe UI", 11),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        pin_var = tk.StringVar(value="")
        pin_digits = []

        pin_display = tk.Label(
            form_frame,
            textvariable=pin_var,
            font=("Segoe UI Semibold", 24, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.ENTRY_BG,
            relief="solid",
            bd=1,
            height=1,
            pady=8,
        )
        pin_display.pack(fill="x", pady=(10, 12))

        keypad_frame = tk.Frame(form_frame, bg=self.PANEL_COLOR)
        keypad_frame.pack(fill="x")
        for column in range(3):
            keypad_frame.grid_columnconfigure(column, weight=1)

        def refresh_pin_display():
            pin_var.set(" ".join("*" for _ in pin_digits))
            return
            pin_var.set(" ".join("●" for _ in pin_digits))

        def clear_pin():
            pin_digits.clear()
            refresh_pin_display()

        def close_admin_login():
            if getattr(self, "admin_panel_frame", None) is login_frame:
                self._close_admin_panel(login_frame)

        def submit_admin_login():
            pin = "".join(pin_digits)
            if len(pin) != 6:
                messagebox.showerror("Loi", "Ma PIN phai co 6 so.", parent=self.root)
                return

            if pin == self.ADMIN_PIN or pin == self.ADMIN_PASSWORD:
                if getattr(self, "admin_panel_frame", None) is login_frame:
                    login_frame.destroy()
                    self.admin_panel_frame = None
                self._clear_admin_login_key_binding()
                self.show_admin_panel()
            else:
                clear_pin()
                messagebox.showerror("Loi", "Ma PIN sai.", parent=self.root)

        def append_pin_digit(digit):
            if len(pin_digits) >= 6:
                return
            pin_digits.append(str(digit))
            refresh_pin_display()
            if len(pin_digits) == 6:
                self.root.after(120, submit_admin_login)

        def backspace_pin():
            if pin_digits:
                pin_digits.pop()
                refresh_pin_display()

        keypad_items = [
            ("1", lambda: append_pin_digit("1")),
            ("2", lambda: append_pin_digit("2")),
            ("3", lambda: append_pin_digit("3")),
            ("4", lambda: append_pin_digit("4")),
            ("5", lambda: append_pin_digit("5")),
            ("6", lambda: append_pin_digit("6")),
            ("7", lambda: append_pin_digit("7")),
            ("8", lambda: append_pin_digit("8")),
            ("9", lambda: append_pin_digit("9")),
            ("Xoa", clear_pin),
            ("0", lambda: append_pin_digit("0")),
            ("Lui", backspace_pin),
        ]

        for index, (label, command) in enumerate(keypad_items):
            is_digit = label.isdigit()
            tk.Button(
                keypad_frame,
                text=label,
                font=("Segoe UI Semibold", 18 if is_digit else 12, "bold"),
                bg=self.SOFT_BUTTON_BG if not is_digit else self.PANEL_COLOR,
                fg=self.TEXT_COLOR,
                activebackground=self.SOFT_BUTTON_ACTIVE_BG,
                relief="solid",
                bd=1,
                cursor="hand2",
                padx=8,
                pady=10,
                command=command,
            ).grid(row=index // 3, column=index % 3, padx=4, pady=4, sticky="nsew")

        hint_label = tk.Label(
            form_frame,
            text="PIN mac dinh: 123456. Co the doi bang bien moi truong LOCKER_ADMIN_PIN.",
            font=("Segoe UI", 8),
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        )
        hint_label.pack(anchor="w", pady=(8, 0))

        button_row = tk.Frame(login_content, bg=self.BG_COLOR)
        button_row.pack(fill="x", padx=20, pady=(12, 18))
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)

        tk.Button(
            button_row,
            text="Dang nhap",
            font=("Segoe UI", 11, "bold"),
            bg=self.PRIMARY_COLOR,
            fg="white",
            activebackground=self.PRIMARY_ACTIVE_BG,
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=10,
            command=submit_admin_login,
        ).grid(row=0, column=0, padx=(0, 10), sticky="ew")

        tk.Button(
            button_row,
            text="Dong",
            font=("Segoe UI", 11, "bold"),
            bg=self.SOFT_BUTTON_BG,
            fg=self.TEXT_COLOR,
            activebackground=self.SOFT_BUTTON_ACTIVE_BG,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=10,
            command=close_admin_login,
        ).grid(row=0, column=1, padx=(10, 0), sticky="ew")

        def handle_pin_keypress(event):
            if event.keysym == "Return":
                submit_admin_login()
                return "break"
            if event.keysym == "Escape":
                close_admin_login()
                return "break"
            if event.keysym == "BackSpace":
                backspace_pin()
                return "break"
            if event.char and event.char.isdigit():
                append_pin_digit(event.char)
                return "break"
            return "break"

        self._clear_admin_login_key_binding()
        self.admin_login_key_bind_id = self.root.bind_all("<KeyPress>", handle_pin_keypress, add="+")
        login_frame.focus_set()

    def show_admin_panel(self):
        existing_admin_frame = getattr(self, "admin_panel_frame", None)
        if existing_admin_frame is not None:
            try:
                existing_admin_frame.destroy()
            except Exception:
                pass
            self.admin_panel_frame = None
        self._clear_admin_login_key_binding()

        self.main_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.fingerprint_frame.pack_forget()
        self.door_frame.pack_forget()

        admin_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=14, pady=14)
        self.admin_panel_frame = admin_frame
        admin_frame.pack(fill="both", expand=True)
        admin_content = self._make_scrollable_page(admin_frame)

        header_frame = tk.Frame(
            admin_content,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        header_frame.pack(fill="x", padx=20, pady=20)
        header_frame.grid_columnconfigure(0, weight=1)

        copy_frame = tk.Frame(header_frame, bg=self.PANEL_COLOR)
        copy_frame.grid(row=0, column=0, sticky="w")

        tk.Label(
            copy_frame,
            text="BANG DIEU KHIEN",
            font=("Segoe UI Semibold", 9, "bold"),
            fg=self.HEADER_ACCENT,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        tk.Label(
            copy_frame,
            text="Quan ly tu do",
            font=("Segoe UI Semibold", 18, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            copy_frame,
            text="Theo doi tung ngan tu, xem thong ke nhanh va xu ly cac tinh huong quan tri.",
            font=self.BODY_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(header_frame, image=self.logo_photo, bg=self.PANEL_COLOR).grid(row=0, column=1, padx=(12, 0), sticky="e")

        auth_mode_frame = tk.Frame(
            admin_content,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=16,
            pady=14,
        )
        auth_mode_frame.pack(fill="x", padx=25, pady=(0, 14))

        tk.Label(
            auth_mode_frame,
            text="Che do xac thuc",
            font=("Segoe UI", 12, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).grid(row=0, column=0, sticky="w")

        tk.Radiobutton(
            auth_mode_frame,
            text="Mode 1: Can ca khuon mat va van tay",
            variable=self.auth_mode_var,
            value=self.AUTH_MODE_FACE_AND_FINGERPRINT,
            command=self._handle_auth_mode_changed,
            font=("Segoe UI", 9),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            selectcolor=self.PANEL_COLOR,
            activebackground=self.PANEL_COLOR,
            anchor="w",
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        tk.Radiobutton(
            auth_mode_frame,
            text="Mode 2: Khuon mat hoac van tay",
            variable=self.auth_mode_var,
            value=self.AUTH_MODE_FACE_OR_FINGERPRINT,
            command=self._handle_auth_mode_changed,
            font=("Segoe UI", 9),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            selectcolor=self.PANEL_COLOR,
            activebackground=self.PANEL_COLOR,
            anchor="w",
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        tk.Radiobutton(
            auth_mode_frame,
            text="Mode 3: Chi khuon mat tren may tinh (khong van tay/relay)",
            variable=self.auth_mode_var,
            value=self.AUTH_MODE_FACE_ONLY_PC,
            command=self._handle_auth_mode_changed,
            font=("Segoe UI", 9),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            selectcolor=self.PANEL_COLOR,
            activebackground=self.PANEL_COLOR,
            anchor="w",
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

        tk.Label(
            auth_mode_frame,
            textvariable=self.fingerprint_status_var,
            font=("Segoe UI", 9),
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

        admin_lockers_frame = tk.Frame(admin_content, bg=self.BG_COLOR)
        admin_lockers_frame.pack(fill="x", padx=25)

        self._populate_admin_lockers(admin_lockers_frame)

        button_panel = tk.Frame(admin_content, bg=self.BG_COLOR)
        button_panel.pack(fill="x", padx=25, pady=(8, 0))
        button_panel.grid_columnconfigure(0, weight=1)
        button_panel.grid_columnconfigure(1, weight=1)
        button_panel.grid_columnconfigure(2, weight=1)

        analytics_button = tk.Button(
            button_panel,
            text="Thong ke",
            font=("Segoe UI", 12, "bold"),
            bg=self.SECONDARY_COLOR,
            fg="white",
            activebackground=self.SECONDARY_ACTIVE_BG,
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=10,
            command=lambda: self.show_analytics_dashboard(admin_frame),
        )
        analytics_button.grid(row=0, column=0, padx=(0, 6), pady=(0, 8), sticky="ew")

        history_button = tk.Button(
            button_panel,
            text="Lich su tu do",
            font=("Segoe UI", 12, "bold"),
            bg=self.PRIMARY_COLOR,
            fg="white",
            activebackground=self.PRIMARY_ACTIVE_BG,
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=10,
            command=lambda: self.show_event_history(admin_frame),
        )
        history_button.grid(row=0, column=1, padx=(6, 0), pady=(0, 8), sticky="ew")

        model_health_button = tk.Button(
            button_panel,
            text="Suc khoe model",
            font=("Segoe UI", 12, "bold"),
            bg=self.INFO_ACCENT,
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=10,
            command=lambda: self.show_model_health_dashboard(admin_frame),
        )
        model_health_button.grid(row=0, column=2, padx=(6, 0), pady=(0, 8), sticky="ew")

        clear_fingerprints_button = tk.Button(
            button_panel,
            text="Xoa toan bo van tay cam bien",
            font=("Segoe UI", 12, "bold"),
            bg=self.DANGER_COLOR,
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=10,
            command=self._clear_all_fingerprints_from_sensor,
        )
        clear_fingerprints_button.grid(row=1, column=0, columnspan=3, pady=(0, 8), sticky="ew")

        shutdown_button = tk.Button(
            button_panel,
            text="Tat phan mem",
            font=("Segoe UI", 12, "bold"),
            bg=self.DANGER_COLOR,
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=10,
            command=lambda: self._confirm_admin_shutdown(admin_frame),
        )
        shutdown_button.grid(row=2, column=0, columnspan=3, pady=(0, 8), sticky="ew")

        close_button = tk.Button(
            button_panel,
            text="Dong",
            font=("Segoe UI", 12, "bold"),
            bg=self.SOFT_BUTTON_BG,
            fg=self.TEXT_COLOR,
            activebackground=self.SOFT_BUTTON_ACTIVE_BG,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=10,
            command=lambda: self._close_admin_panel(admin_frame),
        )
        close_button.grid(row=3, column=0, columnspan=2, pady=(4, 0), sticky="ew")

    def _close_admin_panel(self, admin_frame=None):
        frame = admin_frame or getattr(self, "admin_panel_frame", None)
        if frame is not None:
            try:
                frame.destroy()
            except Exception:
                pass
        self.admin_panel_frame = None
        self._clear_admin_login_key_binding()
        if (
            not self.main_frame.winfo_ismapped()
            and not self.camera_frame.winfo_ismapped()
            and not self.fingerprint_frame.winfo_ismapped()
            and not self.door_frame.winfo_ismapped()
        ):
            self.main_frame.pack(fill="both", expand=True)

    def _hide_admin_panel(self):
        frame = getattr(self, "admin_panel_frame", None)
        if frame is None:
            self._clear_admin_login_key_binding()
            return
        try:
            frame.destroy()
        except Exception:
            pass
        self.admin_panel_frame = None
        self._clear_admin_login_key_binding()

    def _clear_admin_login_key_binding(self):
        bind_id = getattr(self, "admin_login_key_bind_id", None)
        if bind_id is None:
            return
        try:
            self.root.unbind_all("<KeyPress>")
        except Exception:
            pass
        self.admin_login_key_bind_id = None

    def _confirm_admin_shutdown(self, parent_window):
        confirmed = messagebox.askyesno(
            "Xac nhan",
            "Ban co chac chan muon tat phan mem?",
            parent=parent_window,
        )
        if not confirmed:
            return

        try:
            parent_window.destroy()
        except Exception:
            pass
        self.on_close()

    def show_event_history(self, parent_window=None):
        return_to_admin = parent_window is not None
        self.main_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.fingerprint_frame.pack_forget()
        self.door_frame.pack_forget()
        self._hide_admin_panel()

        history_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=14, pady=14)
        self.admin_panel_frame = history_frame
        history_frame.pack(fill="both", expand=True)

        main_frame = tk.Frame(history_frame, bg=self.BG_COLOR, padx=20, pady=24)
        main_frame.pack(fill="both", expand=True)

        header_frame = tk.Frame(
            main_frame,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=16,
            pady=18,
        )
        header_frame.pack(fill="x")
        header_frame.grid_columnconfigure(0, weight=1)

        copy_frame = tk.Frame(header_frame, bg=self.PANEL_COLOR)
        copy_frame.grid(row=0, column=0, sticky="w")

        tk.Label(
            copy_frame,
            text="NHAT KY HOAT DONG",
            font=("Segoe UI Semibold", 9, "bold"),
            fg=self.HEADER_ACCENT,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        tk.Label(
            copy_frame,
            text="Lich su tu do",
            font=("Segoe UI Semibold", 18, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            copy_frame,
            text="Xem lai cac thao tac gui do, lay do va xu ly quan tri gan day.",
            font=("Segoe UI", 11),
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            pady=8,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w")

        tk.Label(header_frame, image=self.logo_photo, bg=self.PANEL_COLOR).grid(row=0, column=1, sticky="e", padx=(12, 0))

        table_wrap = tk.Frame(
            main_frame,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=8,
            pady=8,
        )
        table_wrap.pack(fill="both", expand=True, pady=(20, 16))

        columns = ("thoi_gian", "hanh_dong", "ten_tu", "do_tin_cay", "trang_thai_sau", "anh", "ghi_chu")
        tree = ttk.Treeview(table_wrap, columns=columns, show="headings", height=24)
        headings = {
            "thoi_gian": "Thoi gian",
            "hanh_dong": "Hanh dong",
            "ten_tu": "Ten tu",
            "do_tin_cay": "Do tin cay",
            "trang_thai_sau": "Trang thai",
            "anh": "Anh",
            "ghi_chu": "Ghi chu",
        }
        column_widths = {
            "thoi_gian": 160,
            "hanh_dong": 150,
            "ten_tu": 70,
            "do_tin_cay": 90,
            "trang_thai_sau": 110,
            "anh": 60,
            "ghi_chu": 420,
        }

        for column in columns:
            tree.heading(column, text=headings[column])
            anchor = "center" if column in {"ten_tu", "do_tin_cay", "trang_thai_sau", "anh"} else "w"
            tree.column(column, width=column_widths[column], anchor=anchor, stretch=False)

        table_wrap.grid_rowconfigure(0, weight=1)
        table_wrap.grid_columnconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(table_wrap, orient="vertical", command=tree.yview)
        horizontal_scrollbar = ttk.Scrollbar(table_wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal_scrollbar.set)

        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        rows = self._read_event_history()
        row_metadata = {}
        if rows:
            for row in reversed(rows):
                snapshot_path = self._resolve_event_snapshot_path(row)
                row["__snapshot_path"] = snapshot_path
                item_id = tree.insert(
                    "",
                    "end",
                    values=(
                        row.get("thoi_gian", ""),
                        row.get("hanh_dong", ""),
                        row.get("ten_tu", ""),
                        row.get("do_tin_cay", ""),
                        row.get("trang_thai_sau", "") or self._status_from_history_label(row.get("hanh_dong", "")),
                        "Co" if snapshot_path else "Khong",
                        row.get("ghi_chu", ""),
                    ),
                )
                row_metadata[item_id] = row
        else:
            tree.insert("", "end", values=("", "Chua co lich su", "", "", "", "", ""))

        button_row = tk.Frame(main_frame, bg=self.BG_COLOR)
        button_row.pack(fill="x")
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)

        preview_photo_ref = {"photo": None}
        preview_panel = tk.Frame(
            main_frame,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=12,
            pady=12,
        )

        def hide_snapshot_preview():
            preview_photo_ref["photo"] = None
            for widget in preview_panel.winfo_children():
                widget.destroy()
            preview_panel.pack_forget()
            table_wrap.pack(fill="both", expand=True, pady=(20, 16))
            button_row.pack(fill="x")

        def close_history():
            if getattr(self, "admin_panel_frame", None) is history_frame:
                try:
                    history_frame.destroy()
                except Exception:
                    pass
                self.admin_panel_frame = None
            if return_to_admin:
                self.show_admin_panel()
            elif (
                not self.main_frame.winfo_ismapped()
                and not self.camera_frame.winfo_ismapped()
                and not self.fingerprint_frame.winfo_ismapped()
                and not self.door_frame.winfo_ismapped()
            ):
                self.main_frame.pack(fill="both", expand=True)

        def show_snapshot_preview():
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("Thong bao", "Hay chon mot dong trong lich su.", parent=self.root)
                return

            selected_row = row_metadata.get(selection[0], {})
            snapshot_path = selected_row.get("__snapshot_path") or self._resolve_event_snapshot_path(selected_row)
            if not snapshot_path:
                messagebox.showinfo(
                    "Thong bao",
                    "Su kien nay chua co anh minh chung. Anh chi duoc luu khi thao tac co khung hinh camera.",
                    parent=self.root,
                )
                return

            try:
                preview_image = Image.open(snapshot_path).convert("RGB")
                max_preview_width = max(280, self.window_width_px - 120)
                max_preview_height = max(260, self.window_height_px - 360)
                preview_image.thumbnail((max_preview_width, max_preview_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(preview_image)
            except Exception as exc:
                messagebox.showerror("Loi", f"Khong mo duoc anh: {exc}", parent=self.root)
                return

            preview_photo_ref["photo"] = photo
            for widget in preview_panel.winfo_children():
                widget.destroy()
            table_wrap.pack_forget()
            button_row.pack_forget()
            preview_panel.pack(fill="both", expand=True, pady=(20, 16))

            tk.Label(
                preview_panel,
                text="Anh su kien",
                font=("Segoe UI Semibold", 16, "bold"),
                fg=self.TEXT_COLOR,
                bg=self.PANEL_COLOR,
            ).pack(anchor="w", pady=(0, 12))

            image_label = tk.Label(preview_panel, image=photo, bg=self.PANEL_COLOR)
            image_label.image = photo
            image_label.pack(expand=True, pady=(0, 12))

            tk.Label(
                preview_panel,
                text=os.path.basename(snapshot_path),
                font=("Segoe UI", 10),
                fg=self.MUTED_COLOR,
                bg=self.PANEL_COLOR,
                wraplength=self.content_wrap_px,
                justify="center",
            ).pack(fill="x")

            tk.Button(
                preview_panel,
                text="Tro lai lich su",
                font=("Segoe UI", 11, "bold"),
                bg=self.SOFT_BUTTON_BG,
                fg=self.TEXT_COLOR,
                activebackground=self.SOFT_BUTTON_ACTIVE_BG,
                relief="flat",
                bd=0,
                cursor="hand2",
                padx=14,
                pady=8,
                command=hide_snapshot_preview,
            ).pack(fill="x", pady=(12, 0))

        tk.Button(
            button_row,
            text="Xem anh su kien",
            font=("Segoe UI", 11, "bold"),
            bg=self.PRIMARY_COLOR,
            fg="white",
            activebackground=self.PRIMARY_ACTIVE_BG,
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=8,
            command=show_snapshot_preview,
        ).grid(row=0, column=0, padx=(0, 8), sticky="ew")

        tk.Button(
            button_row,
            text="Quay lai",
            font=("Segoe UI", 11, "bold"),
            bg=self.SOFT_BUTTON_BG,
            fg=self.TEXT_COLOR,
            activebackground=self.SOFT_BUTTON_ACTIVE_BG,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=8,
            command=close_history,
        ).grid(row=0, column=1, padx=(8, 0), sticky="ew")

    def show_analytics_dashboard(self, parent_window=None):
        return_to_admin = parent_window is not None
        self.main_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.fingerprint_frame.pack_forget()
        self.door_frame.pack_forget()
        self._hide_admin_panel()

        dashboard = tk.Frame(self.root, bg=self.BG_COLOR, padx=14, pady=14)
        self.admin_panel_frame = dashboard
        dashboard.pack(fill="both", expand=True)

        summary = self._build_usage_summary()

        header_frame = tk.Frame(
            dashboard,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        header_frame.pack(fill="x", padx=20, pady=20)
        header_frame.grid_columnconfigure(0, weight=1)

        copy_frame = tk.Frame(header_frame, bg=self.PANEL_COLOR)
        copy_frame.grid(row=0, column=0, sticky="w")

        tk.Label(
            copy_frame,
            text="THONG KE HE THONG",
            font=("Segoe UI Semibold", 9, "bold"),
            fg=self.HEADER_ACCENT,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        tk.Label(
            copy_frame,
            text="Thong ke van hanh",
            font=("Segoe UI Semibold", 18, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            copy_frame,
            text="Tong hop nhanh tan suat su dung, xac thuc that bai va canh bao hien tai.",
            font=self.BODY_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(header_frame, image=self.logo_photo, bg=self.PANEL_COLOR).grid(row=0, column=1, sticky="e", padx=(12, 0))

        stats_frame = tk.Frame(dashboard, bg=self.BG_COLOR)
        stats_frame.pack(fill="both", expand=True, padx=20)
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)

        stat_cards = [
            ("Tong su kien", str(summary["total_events"]), self.PRIMARY_COLOR),
            ("Dang ky thanh cong", str(summary["register_success"]), self.SECONDARY_COLOR),
            ("Lay do thanh cong", str(summary["take_success"]), self.INFO_ACCENT),
            ("Xac thuc that bai", str(summary["failed_auth"]), self.DANGER_COLOR),
            ("Tac vu admin", str(summary["admin_actions"]), "#6b7280"),
            ("Canh bao hien tai", summary["active_alert"], self.WARNING_ACCENT),
        ]

        for index, (label_text, value_text, accent_color) in enumerate(stat_cards):
            card = tk.Frame(
                stats_frame,
                bg=self.PANEL_COLOR,
                highlightbackground=self.BORDER_COLOR,
                highlightthickness=1,
                padx=12,
                pady=12,
            )
            card.grid(row=index // 2, column=index % 2, padx=6, pady=6, sticky="nsew")
            tk.Label(
                card,
                text=label_text,
                font=("Segoe UI", 10, "bold"),
                fg=self.MUTED_COLOR,
                bg=self.PANEL_COLOR,
                wraplength=self.card_wrap_px,
                justify="left",
            ).pack(anchor="w")
            tk.Label(
                card,
                text=value_text,
                font=("Segoe UI", 13, "bold"),
                fg=accent_color,
                bg=self.PANEL_COLOR,
                wraplength=self.card_wrap_px,
                justify="left",
            ).pack(anchor="w", pady=(8, 0))

        def close_dashboard():
            if getattr(self, "admin_panel_frame", None) is dashboard:
                try:
                    dashboard.destroy()
                except Exception:
                    pass
                self.admin_panel_frame = None
            if return_to_admin:
                self.show_admin_panel()
            elif (
                not self.main_frame.winfo_ismapped()
                and not self.camera_frame.winfo_ismapped()
                and not self.fingerprint_frame.winfo_ismapped()
                and not self.door_frame.winfo_ismapped()
            ):
                self.main_frame.pack(fill="both", expand=True)

        tk.Button(
            dashboard,
            text="Quay lai",
            font=("Segoe UI", 11, "bold"),
            bg=self.SOFT_BUTTON_BG,
            fg=self.TEXT_COLOR,
            activebackground=self.SOFT_BUTTON_ACTIVE_BG,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=8,
            command=close_dashboard,
        ).pack(fill="x", padx=20, pady=20)

    def show_model_health_dashboard(self, parent_window=None):
        return_to_admin = parent_window is not None
        self.main_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.fingerprint_frame.pack_forget()
        self.door_frame.pack_forget()
        self._hide_admin_panel()

        dashboard = tk.Frame(self.root, bg=self.BG_COLOR, padx=14, pady=14)
        self.admin_panel_frame = dashboard
        dashboard.pack(fill="both", expand=True)

        try:
            health = inspect_model_consistency()
        except Exception as exc:
            health = {
                "missing_in_model": [],
                "missing_in_dataset": [],
                "has_upper_features": False,
                "encoding_count": 0,
                "error": str(exc),
            }

        try:
            separation = inspect_identity_separation(top_k=8)
        except Exception as exc:
            separation = {
                "user_count": 0,
                "base_encoding_user_count": 0,
                "nearest_neighbors": {},
                "risk_pairs": [],
                "error": str(exc),
            }

        header_frame = tk.Frame(
            dashboard,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        header_frame.pack(fill="x", padx=20, pady=20)
        header_frame.grid_columnconfigure(0, weight=1)

        copy_frame = tk.Frame(header_frame, bg=self.PANEL_COLOR)
        copy_frame.grid(row=0, column=0, sticky="w")

        tk.Label(
            copy_frame,
            text="SUC KHOE MODEL",
            font=("Segoe UI Semibold", 9, "bold"),
            fg=self.HEADER_ACCENT,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        tk.Label(
            copy_frame,
            text="Kiem tra do tach biet danh tinh",
            font=("Segoe UI Semibold", 18, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            copy_frame,
            text=(
                "Man hinh nay cho biet cap nao de bi nham nhat trong model. "
                "Neu khoang cach min qua nho, nen chup/train lai du lieu cua cap do."
            ),
            font=self.BODY_FONT,
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            wraplength=self.content_wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(header_frame, image=self.logo_photo, bg=self.PANEL_COLOR).grid(row=0, column=1, sticky="e", padx=(12, 0))

        cards_frame = tk.Frame(dashboard, bg=self.BG_COLOR)
        cards_frame.pack(fill="x", padx=20)
        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)

        high_risk_count = len([row for row in separation.get("risk_pairs", []) if row.get("risk_level") == "high"])
        medium_risk_count = len([row for row in separation.get("risk_pairs", []) if row.get("risk_level") == "medium"])
        card_rows = [
            ("Tong user trong model", str(separation.get("user_count", 0)), self.PRIMARY_COLOR),
            ("Base user co 128-dim", str(separation.get("base_encoding_user_count", 0)), self.INFO_ACCENT),
            ("Tong encodings", str(health.get("encoding_count", 0)), self.SECONDARY_COLOR),
            ("Upper-face feature", "Co" if health.get("has_upper_features") else "Chua", self.WARNING_ACCENT if not health.get("has_upper_features") else self.PRIMARY_COLOR),
            ("Cap nguy co cao", str(high_risk_count), self.DANGER_COLOR),
            ("Cap can theo doi", str(medium_risk_count), self.WARNING_ACCENT),
        ]

        for index, (label_text, value_text, accent_color) in enumerate(card_rows):
            card = tk.Frame(
                cards_frame,
                bg=self.PANEL_COLOR,
                highlightbackground=self.BORDER_COLOR,
                highlightthickness=1,
                padx=12,
                pady=12,
            )
            card.grid(row=index // 2, column=index % 2, padx=6, pady=6, sticky="nsew")
            tk.Label(
                card,
                text=label_text,
                font=("Segoe UI", 10, "bold"),
                fg=self.MUTED_COLOR,
                bg=self.PANEL_COLOR,
                wraplength=self.card_wrap_px,
                justify="left",
            ).pack(anchor="w")
            tk.Label(
                card,
                text=value_text,
                font=("Segoe UI", 13, "bold"),
                fg=accent_color,
                bg=self.PANEL_COLOR,
                wraplength=self.card_wrap_px,
                justify="left",
            ).pack(anchor="w", pady=(8, 0))

        detail_frame = tk.Frame(
            dashboard,
            bg=self.PANEL_COLOR,
            highlightbackground=self.BORDER_COLOR,
            highlightthickness=1,
            padx=16,
            pady=14,
        )
        detail_frame.pack(fill="both", expand=True, padx=20, pady=(12, 0))

        summary_lines = []
        if health.get("missing_in_model"):
            summary_lines.append("Dataset chua train: " + ", ".join(health["missing_in_model"]))
        if health.get("missing_in_dataset"):
            summary_lines.append("Model du nhan: " + ", ".join(health["missing_in_dataset"]))
        if not summary_lines:
            summary_lines.append("Model va dataset dang dong bo o muc co ban.")
        if separation.get("error"):
            summary_lines.append(f"Loi separation: {separation['error']}")
        if health.get("error"):
            summary_lines.append(f"Loi consistency: {health['error']}")

        tk.Label(
            detail_frame,
            text="Tom tat nhanh",
            font=("Segoe UI", 12, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        tk.Label(
            detail_frame,
            text="\n".join(f"- {line}" for line in summary_lines),
            font=("Consolas", 10),
            fg=self.MUTED_COLOR,
            bg=self.PANEL_COLOR,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(8, 12))

        tk.Label(
            detail_frame,
            text="Top cap de nham",
            font=("Segoe UI", 12, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        pair_lines = []
        for row in separation.get("risk_pairs", []):
            pair_lines.append(
                f"- {row['user_a']} <-> {row['user_b']} | min={row['min_distance']:.3f} | "
                f"mean={row['mean_distance']:.3f} | risk={row['risk_level']}"
            )
        if not pair_lines:
            pair_lines.append("- Chua du du lieu de tinh cap de nham.")

        tk.Label(
            detail_frame,
            text="\n".join(pair_lines),
            font=("Consolas", 10),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(8, 12))

        tk.Label(
            detail_frame,
            text="Gan nhat theo tung ID",
            font=("Segoe UI", 12, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
        ).pack(anchor="w")

        neighbor_lines = []
        for user_name in sorted(separation.get("nearest_neighbors", {})):
            neighbor = separation["nearest_neighbors"][user_name]
            neighbor_lines.append(
                f"- {user_name} -> {neighbor['neighbor']} | min={neighbor['min_distance']:.3f}"
            )
        if not neighbor_lines:
            neighbor_lines.append("- Chua co thong tin nearest-neighbor.")

        tk.Label(
            detail_frame,
            text="\n".join(neighbor_lines),
            font=("Consolas", 10),
            fg=self.TEXT_COLOR,
            bg=self.PANEL_COLOR,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

        def close_dashboard():
            if getattr(self, "admin_panel_frame", None) is dashboard:
                try:
                    dashboard.destroy()
                except Exception:
                    pass
                self.admin_panel_frame = None
            if return_to_admin:
                self.show_admin_panel()
            elif (
                not self.main_frame.winfo_ismapped()
                and not self.camera_frame.winfo_ismapped()
                and not self.fingerprint_frame.winfo_ismapped()
                and not self.door_frame.winfo_ismapped()
            ):
                self.main_frame.pack(fill="both", expand=True)

        tk.Button(
            dashboard,
            text="Quay lai",
            font=("Segoe UI", 11, "bold"),
            bg=self.SOFT_BUTTON_BG,
            fg=self.TEXT_COLOR,
            activebackground=self.SOFT_BUTTON_ACTIVE_BG,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=8,
            command=close_dashboard,
        ).pack(fill="x", padx=20, pady=20)

    def _read_event_history(self):
        if not os.path.exists(self.EVENT_LOG_FILE):
            return []

        try:
            with open(self.EVENT_LOG_FILE, "r", newline="", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self._cleanup_unreferenced_event_snapshots(rows)
            return rows
        except Exception:
            return []

    def _action_key_from_history_label(self, action_label):
        action_keys = {
            "Huy gui do": "keep_cancelled",
            "Chan gui do": "keep_blocked",
            "Tu choi gui do": "keep_rejected",
            "Dang ky that bai": "register_failed",
            "Dang ky thanh cong": "register_success",
            "Huy lay do": "take_cancelled",
            "Tu choi lay do": "take_rejected",
            "Lay do tam thoi": "take_temporary",
            "Ket thuc gui do": "take_finish",
            "Admin ep xoa": "admin_force_clear",
            "Admin xoa toan bo van tay": "admin_clear_fingerprints",
        }
        return action_keys.get(action_label, "")

    def _resolve_event_snapshot_path(self, row):
        snapshot_path = str(row.get("anh_su_kien", "") or "").strip()
        for candidate in self._snapshot_path_candidates(snapshot_path):
            if os.path.exists(candidate):
                return candidate
        return self._find_event_snapshot_for_history_row(row)

    def _snapshot_path_candidates(self, snapshot_path):
        if not snapshot_path:
            return []

        candidates = []
        if os.path.isabs(snapshot_path):
            candidates.append(snapshot_path)
        else:
            base_dirs = [
                os.getcwd(),
                os.path.dirname(os.path.abspath(self.EVENT_LOG_FILE)),
                os.path.dirname(os.path.abspath(__file__)),
            ]
            candidates.append(snapshot_path)
            for base_dir in base_dirs:
                candidates.append(os.path.join(base_dir, snapshot_path))

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            normalized = os.path.normpath(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(normalized)
        return unique_candidates

    def _find_event_snapshot_for_history_row(self, row):
        action_key = self._action_key_from_history_label(row.get("hanh_dong", ""))
        if not action_key:
            return ""

        try:
            event_time = datetime.fromisoformat(str(row.get("thoi_gian", "")).strip())
        except ValueError:
            return ""

        timestamp = event_time.strftime("%Y%m%d_%H%M%S")
        locker_name = str(row.get("ten_tu", "") or "unknown")
        snapshot_dirs = self._snapshot_path_candidates(self.SNAPSHOT_DIR)
        for snapshot_dir in snapshot_dirs:
            if not os.path.isdir(snapshot_dir):
                continue

            for extension in (".jpg", ".jpeg", ".png"):
                exact_path = os.path.join(snapshot_dir, f"{timestamp}_{action_key}_{locker_name}{extension}")
                if os.path.exists(exact_path):
                    return exact_path

            prefix = f"{timestamp}_{action_key}_"
            try:
                file_names = os.listdir(snapshot_dir)
            except OSError:
                continue
            for file_name in file_names:
                if not file_name.startswith(prefix):
                    continue
                if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if locker_name and locker_name not in file_name:
                    continue
                return os.path.join(snapshot_dir, file_name)

        return ""

    def _populate_admin_lockers(self, parent_frame):
        for widget in parent_frame.winfo_children():
            widget.destroy()

        self.admin_preview_images.clear()
        used_lockers = self._get_used_locker_ids()
        parent_frame.grid_columnconfigure(0, weight=1)

        for i, locker_id in enumerate(self.LOCKER_IDS):
            locker_frame = tk.Frame(
                parent_frame,
                bg=self.PANEL_COLOR,
                highlightbackground=self.BORDER_COLOR,
                highlightthickness=1,
                padx=15,
                pady=15,
            )
            locker_frame.grid(row=i, column=0, sticky="ew", pady=(0, 10))
            locker_frame.grid_columnconfigure(2, weight=1)

            tk.Label(
                locker_frame, text=f"Tu so {locker_id}", font=("Segoe UI", 14, "bold"),
                fg=self.TEXT_COLOR, bg=self.PANEL_COLOR,
            ).grid(row=0, column=0, sticky="w")

            if locker_id in used_lockers:
                preview_image = self._load_locker_preview(locker_id)
                if preview_image is not None:
                    self.admin_preview_images[locker_id] = preview_image
                    tk.Label(
                        locker_frame,
                        image=preview_image,
                        bg=self.PANEL_COLOR,
                    ).grid(row=0, column=1, rowspan=2, padx=(16, 14), sticky="w")
                else:
                    tk.Label(
                        locker_frame,
                        text="Khong co anh",
                        font=("Segoe UI", 10),
                        fg=self.MUTED_COLOR,
                        bg=self.PANEL_COLOR,
                    ).grid(row=0, column=1, rowspan=2, padx=(16, 14), sticky="w")

                tk.Label(
                    locker_frame, text="DANG SU DUNG", font=("Segoe UI", 12),
                    fg=self.SECONDARY_COLOR, bg=self.PANEL_COLOR,
                ).grid(row=0, column=2, sticky="w")

                tk.Button(
                    locker_frame, text="Ep xoa", font=("Segoe UI", 11, "bold"), bg=self.DANGER_COLOR,
                    fg="white", activebackground="#b91c1c", relief="flat", bd=0, cursor="hand2",
                    command=lambda lid=locker_id, pf=parent_frame: self._force_clear_locker(lid, pf),
                ).grid(row=0, column=3, sticky="e")
            else:
                tk.Label(
                    locker_frame, text="CON TRONG", font=("Segoe UI", 12),
                    fg=self.PRIMARY_COLOR, bg=self.PANEL_COLOR,
                ).grid(row=0, column=1, sticky="w", padx=(20, 0))

    def _force_clear_locker(self, locker_id, admin_lockers_frame):
        if not messagebox.askyesno(
            "Xac nhan",
            f"Ban co chac chan muon ep xoa du lieu cua tu so {locker_id}?\n"
            "He thong se xoa ca khuon mat va mau van tay gan voi tu nay.\n"
            "Hanh dong nay khong the hoan tac.",
        ):
            return
        try:
            delete_user(locker_id)
            fingerprint_note = "bo qua van tay (Mode 3)" if self._is_pc_only_mode() else "van tay khong san sang"
            if not self._is_pc_only_mode() and self._ensure_fingerprint_available():
                deleted_fingerprint = self.fingerprint_controller.delete_locker(locker_id)
                if not deleted_fingerprint:
                    raise RuntimeError(f"Cam bien khong xac nhan xoa mau van tay cua tu {locker_id}.")
                fingerprint_note = "da xoa van tay"

            self._append_event_log(
                "admin_force_clear",
                {
                    "name": locker_id,
                    "backend": "Mode 3 PC" if self._is_pc_only_mode() else self.fingerprint_controller.backend_name,
                },
                f"Admin da ep xoa du lieu tu {locker_id}; {fingerprint_note}",
            )
            self.status_var.set(f"Admin da ep xoa du lieu tu {locker_id} ({fingerprint_note}).")
            messagebox.showinfo(
                "Hoan tat",
                f"Da xoa thanh cong du lieu khuon mat cua tu {locker_id}.\n"
                f"Trang thai van tay: {fingerprint_note}.",
            )
            self.refresh_availability_label()
            self._populate_admin_lockers(admin_lockers_frame)
        except Exception as exc:
            messagebox.showerror("Loi", f"Khong the xoa du lieu: {exc}")

    def _clear_all_fingerprints_from_sensor(self):
        if self._is_pc_only_mode():
            messagebox.showinfo(
                "Mode 3",
                "Mode 3 dang chay tren may tinh va khong dung cam bien van tay.",
            )
            return

        confirmed = messagebox.askyesno(
            "Xac nhan",
            "Ban co chac chan muon xoa TOAN BO mau van tay dang luu trong cam bien?\n"
            "Thao tac nay chi xoa du lieu van tay tren cam bien va khong the hoan tac.",
        )
        if not confirmed:
            return

        try:
            if not self._ensure_fingerprint_available():
                raise RuntimeError(self._fingerprint_unavailable_reason())

            cleared = self.fingerprint_controller.clear_all_templates()
            if not cleared:
                raise RuntimeError("Cam bien khong xac nhan xoa toan bo mau van tay.")

            self._append_event_log(
                "admin_clear_fingerprints",
                {"name": "ALL", "confidence": 0.0, "backend": self.fingerprint_controller.backend_name},
                "Admin da xoa toan bo mau van tay tren cam bien",
            )
            self.status_var.set("Admin da xoa toan bo van tay tren cam bien.")
            self._update_auth_mode_status()
            messagebox.showinfo(
                "Hoan tat",
                "Da xoa toan bo mau van tay tren cam bien.\n"
                "Du lieu khuon mat va dataset trong may van duoc giu nguyen.",
            )
        except Exception as exc:
            messagebox.showerror("Loi", f"Khong the xoa toan bo van tay: {exc}")

    def on_close(self):
        self._cancel_door_auto_return()
        self._cancel_fingerprint_startup_retry()
        try:
            self.stop_camera_stream()
        except Exception:
            pass
        self._hide_admin_panel()
        try:
            self.relay_controller.cleanup()
        except Exception:
            pass
        try:
            self.fingerprint_controller.cleanup()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LockerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
