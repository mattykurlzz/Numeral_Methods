import sympy
import matplotlib.pyplot as plt
import numpy as np

import kursach
import tkinter as tk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.pyplot import subplots
from sympy.parsing.mathematica import parse_mathematica


class interface:
    window = None
    displayed_graphics = []
    width, height = None, None
    dpi = None
    met = None
    answer = []
    imitator_flag = False
    destroyables = []
    colors = {
        "dark": "#282525",
        "tree": "#733B2F",
        "mid": "#A64D2D",
        "light": "#BF471B",
        "orange": "#F24B0F",
    }

    def __init__(self, width, height, met):
        self.met = met
        self.window = tk.Tk()
        self.width, self.height = (width, height)
        self.window.geometry("{}x{}".format(self.width, self.height))
        self.window.config(bg=self.colors["dark"])
        self.dpi = self.window.winfo_fpixels("1c")
        frames = [tk.Frame(self.window) for _ in range(4)]
        texts = [
            "Метод бисекции",
            "МПИ и Ньютон",
            "Решение СЛАУ",
            "Решение системы методом Ньютона",
            "Параметры линейной регрессии",
            "Вычисление интегралов",
            "Решение дифференциальных уравнений",
            "Закрыть окно",
        ]
        calls = [
            lambda: self.Bisect_or_NewtFPI("Bisect"),
            lambda: self.Bisect_or_NewtFPI("FPI-Newton"),
            self.SLAU,
            self.non_linear,
            self.regress,
            self.integrate,
            self.diffs,
            self.window.destroy,
        ]
        self.displayed_graphics = [
            tk.Label(
                self.window,
                text="Выберите задание ниже",
                font=("Andale Mono", self.height // 20),
                bg=self.colors["dark"],
                fg=self.colors["light"],
            ),
            *[
                tk.Button(
                    frames[i % 4],
                    text=texts[i],
                    font=("Andale Mono", 30),
                    # width=int(self.width // 24),
                    height=int(self.height // 7 // 24),
                    padx=0,
                    pady=0,
                    relief="flat",
                    command=calls[i],
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                    highlightthickness=0,
                    bd=0,
                    activebackground=self.colors["light"],
                    activeforeground=self.colors["dark"],
                )
                for i in range(len(texts))
            ],
        ]
        self.displayed_graphics[0].pack(side="top")
        [frame.pack(fill="x", side="top", expand=1) for frame in frames]
        [
            self.displayed_graphics[i].pack(
                side="left", anchor="w", fill="both", expand=1
            )
            for i in range(1, 5)
        ]
        [
            self.displayed_graphics[i].pack(
                side="right", anchor="e", fill="both", expand=1
            )
            for i in range(5, 9)
        ]

        self.window.mainloop()
        print("we've reached the init return")
        return None

    def Bisect_or_NewtFPI(self, type):
        x = sympy.symbols("x")
        newwin = tk.Tk()
        newwin.geometry("{}x{}".format(self.width, self.height))
        frames = [
            tk.Frame(newwin, bg=self.colors["dark"]) for _ in range(2)
        ]  # 0 and 1 are bound to newwin
        frames.append(tk.Frame(frames[1], bg=self.colors["light"]))
        frames.append(tk.Frame(frames[1], bg=self.colors["dark"]))
        # 2 and 3 are bound to frame 1
        [
            frames.append(tk.Frame(frames[2], bg=self.colors["light"]))
            for _ in range(3)
        ]  # 4, 5, 6 are bound to frame 2
        Progname = tk.Label(
            frames[0],
            text="Бисекция" if type == "Bisect" else "Метод Ньютона и МПИ",
            font=("Arial", self.height // 20),
            bg=self.colors["orange"],
        )
        Rootlabel = tk.Label(
            frames[0],
            text="Тут отобразятся найденные корни",
            font=("Arial", self.height // 30),
            fg=self.colors["orange"],
            bg=self.colors["dark"],
        )

        frames[0].pack(fill="x", side="top")
        Progname.pack(
            side="left", padx=self.width // 20, pady=self.width // 40, fill="y"
        )
        Rootlabel.pack(side="left", padx=self.width // 20, expand=1)

        frames[1].pack(fill="both", side="top", expand=1)
        frames[3].pack(side="right", fill="both", expand=1, anchor="se")
        frames[2].pack(
            side="left",
            anchor="w",
            padx=self.width // 40,
            pady=self.width // 40,
            fill="y",
        )
        frames[4].pack(side="top", fill="x")
        frames[5].pack(side="top", fill="x")
        frames[6].pack(side="top", fill="x")

        tk.Label(
            frames[4],
            text="Функция: ",
            font=("Andale Mono", 15),
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40)
        left = tk.Entry(
            frames[4],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        left.pack(side="left")
        tk.Label(
            frames[4],
            text=" = ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            font=("Andale Mono", 15),
        ).pack(side="left")
        right = tk.Entry(
            frames[4],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        right.pack(side="left")
        left.insert(0, "2^x-3*x-2" if type == "Bisect" else "Tan[0.5*x-1.2]")
        right.insert(0, "0" if type == "Bisect" else "x^2-1")

        tk.Label(
            frames[5],
            text="Интервал:",
            bg=self.colors["light"],
            fg=self.colors["dark"],
            font=("Andale Mono", 15),
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40)
        tk.Label(
            frames[5],
            text="[",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            font=("Andale Mono", 15),
        ).pack(side="left")
        a = tk.Entry(
            frames[5],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 200,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        a.pack(side="left")
        tk.Label(
            frames[5],
            text=" ; ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            font=("Andale Mono", 15),
        ).pack(side="left")
        b = tk.Entry(
            frames[5],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 200,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        b.pack(side="left")
        tk.Label(
            frames[5],
            text="]",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            font=("Andale Mono", 15),
        ).pack(side="left", expand=0)
        a.insert(0, "-4" if type == "Bisect" else "-2")
        b.insert(0, "4" if type == "Bisect" else "-1.5")

        tk.Label(
            frames[6],
            text="eps = ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
            font=("Andale Mono", 15),
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40)
        eps = tk.Entry(
            frames[6],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        eps.pack(side="left", expand=0)
        eps.insert(0, "0.001" if type == "Bisect" else "0.001")

        tk.Button(
            frames[2],
            text="Закрыть",
            font=("Andale Mono", 20),
            command=lambda: newwin.destroy(),
            bg=self.colors["light"],
            fg=self.colors["dark"],
            activebackground=self.colors["tree"],
            activeforeground=self.colors["orange"],
            bd=0,
            highlightthickness=0,
        ).pack(side="bottom", anchor="sw", fill="x")

        if type == "Bisect":
            self.destroyables = [
                tk.Button(
                    frames[3],
                    text="Найти корни",
                    font=("Andale Mono", 15),
                    bg=self.colors["light"],
                    fg=self.colors["dark"],
                    activebackground=self.colors["dark"],
                    activeforeground=self.colors["orange"],
                    bd=0,
                    highlightthickness=0,
                    command=lambda: [
                        self.function_imitator(
                            [
                                float(a.get()),
                                float(b.get()),
                                float(eps.get()),
                                parse_mathematica(left.get())
                                - parse_mathematica(right.get()),
                            ],
                            self.met.Half,
                        ),
                        self.makeagraph(
                            float(a.get()),
                            float(b.get()),
                            float(eps.get()),
                            sympy.lambdify(
                                x,
                                parse_mathematica(left.get())
                                - parse_mathematica(right.get()),
                            ),
                            frames[3],
                            self.answer[-1],
                            type="Bisect",
                        ),
                        Rootlabel.configure(text=str(self.answer[0])),
                    ],
                )
            ]
            self.destroyables[-1].pack(expand=1)
        elif type == "FPI-Newton":
            self.destroyables = [
                tk.Button(
                    frames[3],
                    text="Найти корни",
                    font=("Andale Mono", 15),
                    bg=self.colors["light"],
                    fg=self.colors["dark"],
                    activebackground=self.colors["dark"],
                    activeforeground=self.colors["orange"],
                    bd=0,
                    highlightthickness=0,
                    command=lambda: [
                        self.function_imitator(
                            [a.get(), b.get(), eps.get(), left.get(), right.get(), x],
                            self.met.MPI,
                        ),
                        self.function_imitator(
                            [a.get(), b.get(), eps.get(), left.get(), right.get(), x],
                            self.met.Newton,
                        ),
                        self.makeagraph(
                            float(a.get()),
                            float(b.get()),
                            float(eps.get()),
                            sympy.lambdify(
                                x,
                                parse_mathematica(left.get())
                                - parse_mathematica(right.get()),
                            ),
                            frames[3],
                            self.answer,
                            type="FPI",
                        ),
                        Rootlabel.configure(
                            text="Ответ МПИ: {}\nОтвет Ньютона: {}".format(
                                self.answer[0], self.answer[1]
                            ),
                            font=self.height // 50,
                        ),
                        print(self.answer),
                    ],
                )
            ]
            self.destroyables[-1].pack(expand=1)
        newwin.mainloop()

        self.destroyables = []
        self.answer = []
        print("we've reached the bisect return")
        return None

    def SLAU(self):
        newwin = tk.Tk()
        newwin.geometry("{}x{}".format(self.width, self.height))
        frames = [
            tk.Frame(newwin, bg=self.colors["dark"]) for _ in range(2)
        ]  # 0 and 1 are bound to newwin
        frames.append(tk.Frame(frames[1], bg=self.colors["light"]))
        frames.append(tk.Frame(frames[1], bg=self.colors["dark"]))
        [
            frames.append(tk.Frame(frames[2], bg=self.colors["light"]))
            for _ in range(2)
        ]  # 4, 5 are bound to frame 2
        [
            frames.append(tk.Frame(frames[4], bg=self.colors["light"]))
            for _ in range(3)
        ]  # 6, 7 are bound to frame 4
        Vecframes = [
            tk.Frame(frames[5], bg=self.colors["light"]) for _ in range(3)
        ]  # are bound to frame 5
        Progname = tk.Label(
            frames[0],
            text="Решение СЛАУ",
            font=("Arial", self.height // 20),
            fg=self.colors["dark"],
            bg=self.colors["orange"],
        )

        frames[0].pack(fill="x", side="top")
        Progname.pack(
            side="left", padx=self.width // 20, pady=self.width // 40, fill="y"
        )

        frames[1].pack(fill="both", side="top", expand=1)  # interactive part
        frames[2].pack(
            side="left",
            anchor="w",
            padx=self.width // 40,
            pady=self.width // 40,
            fill="both",
        )  # the one that contains all of the inputs
        frames[3].pack(
            side="right", fill="both", expand=1, anchor="e"
        )  # this one contains output and buttons
        frames[4].pack(
            side="left", fill="both"
        )  # this contains left sides of equasions
        frames[5].pack(side="right", fill="y")  # this contains the right side

        frames[6].pack(side="top", fill="x")  # left side line 1
        frames[7].pack(side="top", fill="x")  # left side line 2
        frames[8].pack(side="top", fill="x")  # left side line 3

        Vecframes[0].pack(side="top")
        Vecframes[1].pack(side="top")
        Vecframes[2].pack(side="top")

        X = [[]]
        vec = []

        X[0].append(
            tk.Entry(
                frames[6],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[0][0].pack(side="left", expand=0)
        X[0][0].insert(0, "3")
        tk.Label(
            frames[6],
            text="; ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        X[0].append(
            tk.Entry(
                frames[6],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[0][1].pack(side="left", expand=0)
        X[0][1].insert(0, "-1")
        tk.Label(
            frames[6],
            text="; ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        X[0].append(
            tk.Entry(
                frames[6],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[0][2].pack(side="left", expand=0)
        X[0][2].insert(0, "-1")

        vec.append(
            tk.Entry(
                Vecframes[0],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        tk.Label(
            Vecframes[0], text=" = ", bg=self.colors["dark"], fg=self.colors["orange"]
        ).pack(side="left", expand=1)
        vec[-1].pack(side="left")
        vec[-1].insert(0, "2")

        X.append([])
        X[1].append(
            tk.Entry(
                frames[7],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[1][0].pack(side="left", expand=0)
        X[1][0].insert(0, "1")
        tk.Label(
            frames[7],
            text="; ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        X[1].append(
            tk.Entry(
                frames[7],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[1][1].pack(side="left", expand=0)
        X[1][1].insert(0, "1")
        tk.Label(
            frames[7],
            text="; ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        X[1].append(
            tk.Entry(
                frames[7],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[1][2].pack(side="left", expand=0)
        X[1][2].insert(0, "2")

        vec.append(
            tk.Entry(
                Vecframes[1],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        tk.Label(
            Vecframes[1],
            text=" = ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        vec[-1].pack(side="left")
        vec[-1].insert(0, "3")

        X.append([])
        X[2].append(
            tk.Entry(
                frames[8],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[2][0].pack(side="left", expand=0)
        X[2][0].insert(0, "1")
        tk.Label(
            frames[8],
            text="; ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        X[2].append(
            tk.Entry(
                frames[8],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[2][1].pack(side="left", expand=0)
        X[2][1].insert(0, "6")
        tk.Label(
            frames[8],
            text="; ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        X[2].append(
            tk.Entry(
                frames[8],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        X[2][2].pack(side="left", expand=0)
        X[2][2].insert(0, "-1")

        vec.append(
            tk.Entry(
                Vecframes[2],
                width=self.width // 300,
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
            )
        )
        tk.Label(
            Vecframes[2],
            text=" = ",
            bg=self.colors["dark"],
            fg=self.colors["orange"],
        ).pack(side="left", expand=1)
        vec[-1].pack(side="left")
        vec[-1].insert(0, "0")

        tk.Button(
            frames[4],
            text="Закрыть",
            command=lambda: newwin.destroy(),
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            bd=0,
            highlightthickness=0,
            activebackground=self.colors["light"],
            activeforeground=self.colors["dark"],
        ).pack(side="bottom", fill="x")
        epsframe = tk.Frame(frames[4])
        epsframe.pack(side="bottom", fill="x", pady=self.height // 50)
        tk.Label(
            epsframe,
            text="Epsilon = ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left")
        eps = tk.Entry(
            epsframe,
            width=self.width // 100,
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            bd=0,
            highlightthickness=0,
        )
        eps.pack(side="left", expand=0)
        eps.insert(0, "0.0001")

        self.destroyables = [
            tk.Button(
                frames[3],
                text="R +",
                command=lambda: self.modify_size(X, 1, frames, vec, Vecframes),
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                activebackground=self.colors["light"],
                activeforeground=self.colors["dark"],
            )
        ]
        self.destroyables[-1].pack(side="top", expand=1)
        self.destroyables.append(
            tk.Button(
                frames[3],
                text="R -",
                command=lambda: self.modify_size(X, 0, frames, vec, Vecframes),
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                activebackground=self.colors["light"],
                activeforeground=self.colors["dark"],
            )
        )
        self.destroyables[-1].pack(side="top", expand=1)
        self.destroyables.append(
            tk.Button(
                frames[3],
                text="Рассчитать",
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                activebackground=self.colors["light"],
                activeforeground=self.colors["dark"],
                command=lambda: [
                    self.function_imitator(
                        [
                            np.array(
                                [[float(ent.get()) for ent in element] for element in X]
                            ),
                            np.array([float(element.get()) for element in vec]),
                            float(eps.get()),
                        ],
                        met.MatrixMPI,
                    ),
                    tk.Label(
                        frames[3],
                        text="Решение методом МПИ: {}\n времени потрачено - {}, итераций пройдено  - {}\n".format(
                            self.answer[-1], met.time, met.iterations_num
                        ),
                        bg=self.colors["dark"],
                        fg=self.colors["orange"],
                    ).pack(expand=True, side="top"),
                    self.function_imitator(
                        [
                            np.array(
                                [[float(ent.get()) for ent in element] for element in X]
                            ),
                            np.array([float(element.get()) for element in vec]),
                            float(eps.get()),
                        ],
                        met.MatrixZeid,
                    ),
                    tk.Label(
                        frames[3],
                        text="Решение методом Зейделя: {}\n времени потрачено - {}, итераций пройдено  - {}\n".format(
                            self.answer[-1], met.time, met.iterations_num
                        ),
                        bg=self.colors["dark"],
                        fg=self.colors["orange"],
                    ).pack(expand=True, side="top"),
                    [el.destroy() for el in self.destroyables],
                ],
            )
        )
        # print([ent.get() for ent in element]) for element in X
        self.destroyables[-1].pack(side="top", expand=1)

        newwin.mainloop()

        self.destroyables = []
        self.answer = []

        return None

    def non_linear(self):
        newwin = tk.Tk()
        newwin.geometry("{}x{}".format(self.width, self.height))
        super_frames = [
            tk.Frame(newwin, bg=self.colors["dark"]) for _ in range(2)
        ]  # 0 and 1 are bound to newwin
        super_frames[0].pack(fill="x", side="top")
        super_frames[1].pack(fill="both", side="top", expand=1)
        interactive_frames = [
            tk.Frame(super_frames[1], bg=color)
            for color in [self.colors["light"], self.colors["dark"]]
        ]
        interactive_frames[0].pack(side="left", fill="y")
        interactive_frames[1].pack(side="right", fill="both", expand=1)
        equality_frames = [
            tk.Frame(interactive_frames[0], bg=self.colors["dark"]) for _ in range(2)
        ]
        [element.pack(side="top") for element in equality_frames]

        tk.Label(
            super_frames[0],
            text="Решение нелинейных систем",
            font=("Arial", self.height // 20),
            bg=self.colors["orange"],
            fg=self.colors["dark"],
        ).pack(side="left", padx=self.width // 20, pady=self.width // 40, fill="y")

        left_eq = []
        right_eq = []
        basic_ls = ["Sin[x-y]-x*y+1", "x^2-y^2"]
        basic_rs = ["0", "0.75"]

        for i in range(2):
            left_eq.append(
                tk.Entry(
                    equality_frames[i],
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                    highlightthickness=0,
                    bd=0,
                    width=self.width // 50,
                    font=("Andale Mono", 15),
                    insertbackground="white",
                )
            )
            left_eq[i].pack(side="left")
            tk.Label(
                equality_frames[i],
                text=" = ",
                bg=self.colors["dark"],
                fg=self.colors["orange"],
            ).pack(side="left")
            right_eq.append(
                tk.Entry(
                    equality_frames[i],
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                    highlightthickness=0,
                    bd=0,
                    width=self.width // 100,
                    font=("Andale Mono", 15),
                    insertbackground="white",
                )
            )
            right_eq[i].pack(side="left")
            left_eq[i].insert(0, basic_ls[i])
            right_eq[i].insert(0, basic_rs[i])

        epsframe = tk.Frame(interactive_frames[0], bg=self.colors["light"])
        epsframe.pack(side="bottom", fill="x", pady=self.height // 50)
        tk.Label(
            epsframe,
            text="Epsilon = ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left")
        eps = tk.Entry(
            epsframe,
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 200,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        eps.pack(side="left", expand=0)
        eps.insert(0, "0.0001")

        var_frame = tk.Frame(interactive_frames[0], bg=self.colors["light"])
        var_frame.pack(side="bottom", fill="x", pady=self.height // 50)
        tk.Label(
            var_frame,
            text="Vars = ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left")
        vars = tk.Entry(
            var_frame,
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        vars.pack(side="left", expand=0)
        vars.insert(0, "x y")

        self.destroyables = [
            tk.Button(
                interactive_frames[1],
                text="Посчитать",
                font=("Andale Mono", 15),
                bg=self.colors["light"],
                fg=self.colors["dark"],
                activebackground=self.colors["dark"],
                activeforeground=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                command=lambda: [
                    self.function_imitator(
                        [
                            [leq.get() for leq in left_eq],
                            [req.get() for req in right_eq],
                            vars.get(),
                            eps.get(),
                        ],
                        met.NewtonMatrix,
                    ),
                    tk.Label(
                        interactive_frames[1],
                        text="Решение методом Ньютона: {}\n времени потрачено - {}, итераций пройдено  - {}\n".format(
                            self.answer[-1], met.time, met.iterations_num
                        ),
                        fg=self.colors["orange"],
                        bg=self.colors["dark"],
                    ).pack(expand=1),
                    self.function_imitator(
                        [
                            [leq.get() for leq in left_eq],
                            [req.get() for req in right_eq],
                            vars.get(),
                            eps.get(),
                        ],
                        met.MPInonlinear,
                    ),
                    [element.destroy for element in self.destroyables],
                    tk.Label(
                        interactive_frames[1],
                        text="Решение методом простой итерации: {}\n времени потрачено - {}, итераций пройдено  - {}\n".format(
                            self.answer[-1], met.time, met.iterations_num
                        ),
                        fg=self.colors["orange"],
                        bg=self.colors["dark"],
                    ).pack(expand=1),
                ],
            )
        ]
        self.destroyables[-1].pack(side="top", expand=1)

        self.destroyables.append(
            tk.Button(
                interactive_frames[1],
                text="R +",
                font=("Andale Mono", 15),
                bg=self.colors["light"],
                fg=self.colors["dark"],
                activebackground=self.colors["dark"],
                activeforeground=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                command=lambda: [
                    equality_frames.append(tk.Frame(interactive_frames[0], bg="red")),
                    equality_frames[-1].pack(side="top"),
                    left_eq.append(tk.Entry(equality_frames[-1])),
                    left_eq[-1].pack(side="left"),
                    tk.Label(equality_frames[-1], text=" = ").pack(side="left"),
                    right_eq.append(tk.Entry(equality_frames[-1])),
                    right_eq[-1].pack(side="left"),
                ],
            )
        )
        self.destroyables[-1].pack(side="top", expand=1)
        self.destroyables.append(
            tk.Button(
                interactive_frames[1],
                text="R -",
                font=("Andale Mono", 15),
                bg=self.colors["light"],
                fg=self.colors["dark"],
                activebackground=self.colors["dark"],
                activeforeground=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                command=lambda: [
                    equality_frames[-1].destroy(),
                    equality_frames.pop(-1),
                    left_eq.pop(-1),
                    right_eq.pop(-1),
                ],
            )
        )
        self.destroyables[-1].pack(side="top", expand=1)

        newwin.mainloop()
        self.destroyables = []
        self.answer = []

        return None

    def regress(self):
        newwin = tk.Tk()
        newwin.geometry("{}x{}".format(self.width, self.height))
        super_frames = [
            tk.Frame(newwin, bg=self.colors["dark"]) for _ in range(2)
        ]  # 0 and 1 are bound to newwin
        super_frames[0].pack(fill="x", side="top")
        super_frames[1].pack(fill="both", side="top", expand=1)
        interactive_frames = [
            tk.Frame(super_frames[1], bg=color)
            for color in [self.colors["light"], self.colors["dark"]]
        ]
        interactive_frames[0].pack(side="left", fill="y")
        interactive_frames[1].pack(side="right", fill="both", expand=1)
        row_frames = [
            tk.Frame(interactive_frames[0], bg=self.colors["dark"]) for _ in range(10)
        ]
        [element.pack(side="top") for element in row_frames]

        tk.Label(
            super_frames[0],
            text="Параметр линейной регрессии",
            font=("Arial", self.height // 20),
            bg=self.colors["orange"],
            fg=self.colors["dark"],
        ).pack(
            side="left",
            padx=self.width // 40,
            pady=self.width // 40,
            fill="y",
        )
        answer_frame = tk.Frame(super_frames[0], bg=self.colors["dark"])
        answer_frame.pack(side="right", fill="both")

        X = []
        Y = []
        basic_X = [str(num + 1) for num in range(10)]
        basic_Y = ["178", "182", "190", "199", "200", "213", "220", "231", "235", "242"]

        for i in range(10):
            X.append(
                tk.Entry(
                    row_frames[i],
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                    highlightthickness=0,
                    bd=0,
                    width=self.width // 100,
                    font=("Andale Mono", 15),
                    insertbackground="white",
                )
            )
            X[i].pack(side="left")
            tk.Label(
                row_frames[i],
                text=" = ",
                bg=self.colors["dark"],
                fg=self.colors["orange"],
            ).pack(side="left")
            Y.append(
                tk.Entry(
                    row_frames[i],
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                    highlightthickness=0,
                    bd=0,
                    width=self.width // 100,
                    font=("Andale Mono", 15),
                    insertbackground="white",
                )
            )
            Y[i].pack(side="left")
            X[i].insert(0, basic_X[i])
            Y[i].insert(0, basic_Y[i])

        eq_supframe = tk.Frame(interactive_frames[0])
        eq_supframe.pack(side="bottom", fill="x")
        eq_frames = [tk.Frame(eq_supframe, bg=self.colors["light"]) for _ in range(2)]
        eqs = []
        vars = []
        [frame.pack(side="bottom", fill="x") for frame in eq_frames]

        tk.Label(
            eq_frames[0], text="y = ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left")
        eqs.append(
            tk.Entry(
                eq_frames[0],
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                highlightthickness=0,
                bd=0,
                width=self.width // 50,
                font=("Andale Mono", 15),
                insertbackground="white",
            )
        )
        eqs[0].pack(side="left")
        eqs[0].insert(0, "k*x+b")
        tk.Label(
            eq_frames[0],
            text="; coefs = ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left")
        vars.append(
            tk.Entry(
                eq_frames[0],
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                highlightthickness=0,
                bd=0,
                width=self.width // 100,
                font=("Andale Mono", 15),
                insertbackground="white",
            )
        )
        vars[0].pack(side="left", expand=0)
        vars[0].insert(0, "k b")

        tk.Label(
            eq_frames[1], text="y = ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left")
        eqs.append(
            tk.Entry(
                eq_frames[1],
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                highlightthickness=0,
                bd=0,
                width=self.width // 50,
                font=("Andale Mono", 15),
                insertbackground="white",
            )
        )
        eqs[1].pack(side="left")
        eqs[1].insert(0, "a3*x^2+a2*x + a1")
        tk.Label(
            eq_frames[1],
            text="; coefs = ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left")
        vars.append(
            tk.Entry(
                eq_frames[1],
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                highlightthickness=0,
                bd=0,
                width=self.width // 100,
                font=("Andale Mono", 15),
                insertbackground="white",
            )
        )
        vars[1].pack(side="left", expand=0)
        vars[1].insert(0, "a3 a2 a1")

        show_lagrange = tk.IntVar(value=0)
        tk.Checkbutton(
            interactive_frames[0],
            text="Не показывать Лагранжа",
            variable=show_lagrange,
            font=("Arial", self.height // 40),
            bg=self.colors["light"],
            fg=self.colors["dark"],
            activebackground=self.colors["dark"],
            activeforeground=self.colors["orange"],
            borderwidth=0,
            highlightthickness=0,
        ).pack(side="bottom")
        show_newton = tk.IntVar(value=0)
        tk.Checkbutton(
            interactive_frames[0],
            text="Не показывать Ньютона",
            variable=show_newton,
            font=("Arial", self.height // 40),
            bg=self.colors["light"],
            fg=self.colors["dark"],
            activebackground=self.colors["dark"],
            activeforeground=self.colors["orange"],
            borderwidth=0,
            highlightthickness=0,
        ).pack(side="bottom")

        self.destroyables = [
            tk.Button(
                interactive_frames[1],
                text="R +",
                command=lambda: [
                    row_frames.append(tk.Frame(interactive_frames[0], bg="red")),
                    row_frames[-1].pack(side="top"),
                    X.append(tk.Entry(row_frames[-1])),
                    X[-1].pack(side="left"),
                    tk.Label(row_frames[-1], text=" = ").pack(side="left"),
                    Y.append(tk.Entry(row_frames[-1])),
                    Y[-1].pack(side="left"),
                ],
                font=("Arial", self.height // 40),
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                activebackground=self.colors["orange"],
                activeforeground=self.colors["dark"],
                borderwidth=0,
                highlightthickness=0,
            )
        ]
        self.destroyables[-1].pack(side="top", expand=1)
        self.destroyables.append(
            tk.Button(
                interactive_frames[1],
                text="R -",
                command=lambda: [
                    row_frames[-1].destroy(),
                    row_frames.pop(-1),
                    X.pop(-1),
                    Y.pop(-1),
                ],
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                activebackground=self.colors["orange"],
                activeforeground=self.colors["dark"],
                borderwidth=0,
                highlightthickness=0,
            )
        )
        self.destroyables[-1].pack(side="top", expand=1)

        self.destroyables.append(
            tk.Button(
                interactive_frames[1],
                text="Уравнений -",
                command=lambda: [
                    eq_frames[-1].destroy(),
                    eq_frames.pop(-1),
                    eqs.pop(-1),
                    vars.pop(-1),
                ],
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                activebackground=self.colors["orange"],
                activeforeground=self.colors["dark"],
                borderwidth=0,
                highlightthickness=0,
            )
        )
        self.destroyables[-1].pack(side="top", expand=1)
        self.destroyables.append(
            tk.Button(
                interactive_frames[1],
                text="Уравнений +",
                command=lambda: [
                    eq_frames.append(tk.Frame(eq_supframe)),
                    eq_frames[-1].pack(side="bottom", fill="x"),
                    tk.Label(eq_frames[-1], text="y = ").pack(side="left"),
                    eqs.append(tk.Entry(eq_frames[-1])),
                    eqs[-1].pack(side="left"),
                    tk.Label(eq_frames[-1], text="; coefs = ").pack(side="left"),
                    vars.append(tk.Entry(eq_frames[-1])),
                    vars[-1].pack(side="left", expand=0),
                ],
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                activebackground=self.colors["orange"],
                activeforeground=self.colors["dark"],
                borderwidth=0,
                highlightthickness=0,
            )
        )
        self.destroyables[-1].pack(side="top", expand=1)

        self.destroyables.append(
            tk.Button(
                interactive_frames[1],
                text="Посчитать",
                bg=self.colors["dark"],
                fg=self.colors["orange"],
                activebackground=self.colors["orange"],
                activeforeground=self.colors["dark"],
                borderwidth=0,
                highlightthickness=0,
                command=lambda: [
                    [
                        self.function_imitator(
                            [
                                [
                                    [
                                        float(element[i].get())
                                        for i in range(len(element))
                                    ]
                                    for element in [X, Y]
                                ],
                                sympy.symbols(vars[i].get()),
                                eqs[i].get(),
                            ],
                            self.met.Square_inter,
                        )
                        for i in range(len(eqs))
                    ],
                    [
                        tk.Label(
                            answer_frame,
                            text="Получено: {}".format(answer),
                            bg=self.colors["dark"],
                            fg=self.colors["orange"],
                        ).pack(expand=1)
                        for answer in self.answer
                    ],
                    (
                        self.function_imitator(
                            [
                                [
                                    [
                                        float(element[i].get())
                                        for i in range(len(element))
                                    ]
                                    for element in [X, Y]
                                ],
                            ],
                            self.met.Lagrange_inter,
                        )
                        if show_lagrange.get() == 0
                        else None
                    ),
                    (
                        self.function_imitator(
                            [
                                [
                                    [
                                        float(element[i].get())
                                        for i in range(len(element))
                                    ]
                                    for element in [X, Y]
                                ],
                            ],
                            self.met.Newtons_iter,
                        )
                        if show_newton.get() == 0
                        else None
                    ),
                    self.makeagraph(
                        float(X[0].get()),
                        float(X[-1].get()),
                        0.001,
                        self.answer,
                        interactive_frames[1],
                        [
                            [float(element[i].get()) for i in range(len(element))]
                            for element in [X, Y]
                        ],
                        type="regress",
                    ),
                ],
            )
        )
        self.destroyables[-1].pack(side="top", expand=1)
        newwin.mainloop()
        self.destroyables = []
        self.answer = []

        return None

    def integrate(self):
        x = sympy.symbols("x")
        newwin = tk.Tk()
        newwin.geometry("{}x{}".format(self.width, self.height))
        frames = [
            tk.Frame(newwin, bg=self.colors["dark"]) for _ in range(2)
        ]  # 0 and 1 are bound to newwin
        [
            frames.append(tk.Frame(frames[1], bg=color))
            for color in [self.colors["light"], self.colors["dark"]]
        ]  # 2 and 3 are bound to frame 1
        [
            frames.append(tk.Frame(frames[2], bg=self.colors["light"]))
            for _ in range(4)
        ]  # 4, 5, 6, 7 are bound to frame 2
        """
        0 - top side fill x
        1 - top side fill both (actually a bottom half)
        2 - left half of frame1
        3 - right half of frame1 
        4, 5, 6, 7 - three parts of frame 2 
        """
        Progname = tk.Label(
            frames[0],
            text="Интегрирование",
            font=("Arial", self.height // 20),
            bg=self.colors["orange"],
            fg=self.colors["dark"],
        )
        Rootlabel = tk.Label(
            frames[0],
            text="Тут должны были быть найденные корни",
            font=("Arial", self.height // 30),
            bg=self.colors["orange"],
            fg=self.colors["dark"],
        )

        frames[0].pack(fill="x", side="top")
        Progname.pack(
            side="left", padx=self.width // 20, pady=self.width // 40, fill="y"
        )
        Rootlabel.pack(side="left", padx=self.width // 20, expand=1)

        frames[1].pack(fill="both", side="top", expand=1)
        frames[3].pack(side="right", fill="both", expand=1, anchor="se")
        frames[2].pack(
            side="left",
            anchor="w",
            padx=self.width // 40,
            pady=self.width // 40,
            fill="y",
        )
        frames[4].pack(side="top", fill="x")
        frames[5].pack(side="top", fill="x")
        frames[6].pack(side="top", fill="x")
        frames[7].pack(side="top", fill="x")

        tk.Label(
            frames[4], text="Функция: ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        int_function = tk.Entry(
            frames[4],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 30,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        int_function.pack(side="left", expand=1)
        int_function.insert(0, "Log[n, 2]/(4*x^2+3*x+0.4*n)^0.5")

        tk.Label(
            frames[5],
            text="Интервал: [",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        a = tk.Entry(
            frames[5],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        a.pack(side="left", expand=1)
        tk.Label(
            frames[5], text=" ; ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        b = tk.Entry(
            frames[5],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        b.pack(side="left", expand=1)
        tk.Label(
            frames[5], text="]", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", expand=1)
        a.insert(0, "1.2")
        b.insert(0, "2.8")

        tk.Label(
            frames[6], text="eps = ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        eps = tk.Entry(
            frames[6],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        eps.pack(side="left", expand=1)
        eps.insert(0, "0.0001")

        tk.Label(
            frames[7],
            text="переменные кроме x: ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        other_symbols_entry = tk.Entry(
            frames[7],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        other_symbols_entry.pack(side="left", expand=1)
        tk.Label(
            frames[7],
            text="их значения: ",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        other_values_entry = tk.Entry(
            frames[7],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        other_values_entry.pack(side="left", expand=1)
        other_symbols_entry.insert(0, "n")
        other_values_entry.insert(0, "2")

        tk.Button(
            frames[2],
            text="Закрыть",
            font=("Andale Mono", 20),
            command=lambda: newwin.destroy(),
            bg=self.colors["light"],
            fg=self.colors["dark"],
            activebackground=self.colors["tree"],
            activeforeground=self.colors["orange"],
            bd=0,
            highlightthickness=0,
        ).pack(side="bottom", anchor="sw", fill="x")

        self.destroyables = [
            tk.Button(
                frames[3],
                text="найти решение",
                font=("Andale Mono", 15),
                bg=self.colors["light"],
                fg=self.colors["dark"],
                activebackground=self.colors["dark"],
                activeforeground=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                command=lambda: [
                    self.function_imitator(
                        [
                            parse_mathematica(int_function.get()),
                            x,
                            sympy.symbols(other_symbols_entry.get()),
                            list(map(float, other_values_entry.get().split(","))),
                            tuple([float(a.get()), float(b.get()) + float(eps.get())]),
                            float(eps.get()),
                            "squares",
                        ],
                        self.met.Integrate,
                    ),
                    self.function_imitator(
                        [
                            parse_mathematica(int_function.get()),
                            x,
                            sympy.symbols(other_symbols_entry.get()),
                            list(map(float, other_values_entry.get().split(","))),
                            tuple([float(a.get()), float(b.get()) + float(eps.get())]),
                            float(eps.get()),
                            "trapezes",
                        ],
                        self.met.Integrate,
                    ),
                    self.function_imitator(
                        [
                            parse_mathematica(int_function.get()),
                            x,
                            sympy.symbols(other_symbols_entry.get()),
                            list(map(float, other_values_entry.get().split(","))),
                            tuple([float(a.get()), float(b.get()) + float(eps.get())]),
                            float(eps.get()),
                            "Simpson",
                        ],
                        self.met.Integrate,
                    ),
                    self.function_imitator(
                        [
                            parse_mathematica(int_function.get()),
                            x,
                            sympy.symbols(other_symbols_entry.get()),
                            list(map(float, other_values_entry.get().split(","))),
                            tuple([float(a.get()), float(b.get())]),
                            float(eps.get()),
                            "True",
                        ],
                        self.met.Integrate,
                    ),
                    Rootlabel.configure(
                        text="Метод прямоуголькинов: {}; Метод трапеций: {}\nМетод Симпсона: {}; Точный ответ: {}".format(
                            round(self.answer[0], 3),
                            round(self.answer[1], 3),
                            round(self.answer[2], 3),
                            round(self.answer[3], 3),
                        ),
                        font=self.height // 50,
                    ),
                    self.makeabar(
                        frames[3],
                    ),
                ],
            )
        ]
        self.destroyables[-1].pack(expand=1)

        newwin.mainloop()

        self.destroyables = []
        self.answer = []

        return None

    def diffs(self):
        x, y = sympy.symbols("x, y")
        newwin = tk.Tk()
        newwin.geometry("{}x{}".format(self.width, self.height))
        frames = [
            tk.Frame(newwin, bg=self.colors["dark"]) for _ in range(2)
        ]  # 0 and 1 are bound to newwin
        [
            frames.append(tk.Frame(frames[1], bg=color))
            for color in [self.colors["light"], self.colors["dark"]]
        ]  # 2 and 3 are bound to frame 1
        [
            frames.append(tk.Frame(frames[2], bg=self.colors["light"]))
            for _ in range(4)
        ]  # 4, 5, 6, 7 are bound to frame 2
        """
        0 - top side fill x
        1 - top side fill both (actually a bottom half)
        2 - left half of frame1
        3 - right half of frame1 
        4, 5, 6, 7 - three parts of frame 2 
        """
        Progname = tk.Label(
            frames[0],
            text="Дифференцирование",
            font=("Arial", self.height // 20),
            bg=self.colors["orange"],
            fg=self.colors["dark"],
        )
        Rootlabel = tk.Label(
            frames[0],
            text="Тут должны были быть найденные корни",
            font=("Arial", self.height // 30),
            bg=self.colors["orange"],
            fg=self.colors["dark"],
        )

        frames[0].pack(fill="x", side="top")
        Progname.pack(
            side="left", padx=self.width // 20, pady=self.width // 40, fill="y"
        )
        Rootlabel.pack(side="left", padx=self.width // 20, expand=1)

        frames[1].pack(fill="both", side="top", expand=1)
        frames[3].pack(side="right", fill="both", expand=1, anchor="se")
        frames[2].pack(
            side="left",
            anchor="w",
            padx=self.width // 40,
            pady=self.width // 40,
            fill="y",
        )
        frames[4].pack(side="top", fill="x")
        frames[5].pack(side="top", fill="x")
        frames[6].pack(side="top", fill="x")
        frames[7].pack(side="top", fill="x")

        tk.Label(
            frames[4], text="Функция: ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        def_function = tk.Entry(
            frames[4],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        def_function.pack(side="left", expand=1)
        def_function.insert(0, "6 * x^2 + 5 * x * y")

        tk.Label(
            frames[5],
            text="Интервал: [",
            bg=self.colors["light"],
            fg=self.colors["dark"],
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        a = tk.Entry(
            frames[5],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        a.pack(side="left", expand=1)
        tk.Label(
            frames[5], text=" ; ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        b = tk.Entry(
            frames[5],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        b.pack(side="left", expand=1)
        tk.Label(
            frames[5], text="]", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", expand=1)
        a.insert(0, "0")
        b.insert(0, "1")

        tk.Label(
            frames[6], text="h = ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        h = tk.Entry(
            frames[6],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        h.pack(side="left", expand=1)
        h.insert(0, "0.05")

        tk.Label(
            frames[7], text="x0 = ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        x0 = tk.Entry(
            frames[7],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        x0.pack(side="left", expand=1)
        tk.Label(
            frames[7], text="y0 = ", bg=self.colors["light"], fg=self.colors["dark"]
        ).pack(side="left", padx=self.width // 40, pady=self.width // 40, expand=1)
        y0 = tk.Entry(
            frames[7],
            bg=self.colors["dark"],
            fg=self.colors["orange"],
            highlightthickness=0,
            bd=0,
            width=self.width // 100,
            font=("Andale Mono", 15),
            insertbackground="white",
        )
        y0.pack(side="left", expand=1)
        x0.insert(0, "0")
        y0.insert(0, "0")

        tk.Button(
            frames[2],
            text="Закрыть",
            font=("Andale Mono", 20),
            command=lambda: newwin.destroy(),
            bg=self.colors["light"],
            fg=self.colors["dark"],
            activebackground=self.colors["tree"],
            activeforeground=self.colors["orange"],
            bd=0,
            highlightthickness=0,
        ).pack(side="bottom", anchor="sw", fill="x")

        self.destroyables = [
            tk.Button(
                frames[3],
                text="найти решение",
                font=("Andale Mono", 15),
                bg=self.colors["light"],
                fg=self.colors["dark"],
                activebackground=self.colors["dark"],
                activeforeground=self.colors["orange"],
                bd=0,
                highlightthickness=0,
                command=lambda: [
                    self.function_imitator(
                        [
                            parse_mathematica(def_function.get()),
                            x,
                            y,
                            float(x0.get()),
                            float(y0.get()),
                            float(b.get()),
                            float(h.get()),
                        ],
                        self.met.Euler_dif,
                    ),
                    self.function_imitator(
                        [
                            parse_mathematica(def_function.get()),
                            x,
                            y,
                            float(x0.get()),
                            float(y0.get()),
                            float(b.get()),
                            float(h.get()),
                        ],
                        self.met.Runge_4th,
                    ),
                    self.function_imitator(
                        [
                            parse_mathematica(def_function.get()),
                            x,
                            y,
                            float(x0.get()),
                            float(y0.get()),
                            float(b.get()),
                            float(h.get()),
                        ],
                        self.met.diff_analytic_solve,
                    ),
                    Rootlabel.configure(
                        text="Решено!\nАналитический ответ:{}".format(self.answer[2]),
                        font=("Andale Mono", 10),
                    ),
                    self.makeagraph(
                        float(a.get()),
                        float(b.get()),
                        float(h.get()),
                        self.answer,
                        frames[3],
                        0,
                        "diff",
                    ),
                ],
            )
        ]
        self.destroyables[-1].pack(expand=1)

        newwin.mainloop()

        self.destroyables = []
        self.answer = []

        return None

    def function_imitator(self, args, function):
        self.imitator_flag = True
        self.answer.append(function(*args))
        return None

    def modify_size(self, X, type, frames, vec, vecframes):
        if type == 1:
            for i in range(len(X)):
                tk.Label(
                    frames[6 + i],
                    text="; ",
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                ).pack(side="left", expand=1)
                X[i].append(
                    tk.Entry(
                        frames[6 + i],
                        width=self.width // 300,
                        bg=self.colors["dark"],
                        fg=self.colors["orange"],
                        bd=0,
                        highlightthickness=0,
                    )
                )
                X[i][-1].pack(side="left", expand=0)
            X.append([])
            frames.append(tk.Frame(frames[4], bg=self.colors["dark"]))
            frames[-1].pack(side="top", fill="x")
            for i in range(len(X[0])):
                X[-1].append(
                    tk.Entry(
                        frames[-1],
                        width=self.width // 300,
                        bg=self.colors["dark"],
                        fg=self.colors["orange"],
                        bd=0,
                        highlightthickness=0,
                    )
                )
                X[-1][-1].pack(side="left", expand=0)
                tk.Label(
                    frames[-1],
                    text="; ",
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                ).pack(side="left", expand=1)

            vecframes.append(tk.Frame(frames[5], bg="green"))
            vecframes[-1].pack(side="top")
            tk.Label(
                vecframes[-1],
                text=" = ",
                bg=self.colors["dark"],
                fg=self.colors["orange"],
            ).pack(expand=True, side="left")
            vec.append(
                tk.Entry(
                    vecframes[-1],
                    width=self.width // 300,
                    bg=self.colors["dark"],
                    fg=self.colors["orange"],
                    bd=0,
                    highlightthickness=0,
                )
            )
            vec[-1].pack(side="left")

        elif type == 0:
            for i in range(len(X)):
                X[i][-1].destroy()
                X[i].pop(-1)
            for i in range(len(X[0])):
                X[-1][i].destroy()
            frames[-1].destroy()
            frames.pop(-1)
            X.pop(-1)
            vecframes[-1].destroy()
            vecframes.pop(-1)
            vec.pop(-1)

    def makeabar(self, frame):
        fig, ax = subplots(
            figsize=(
                frame.winfo_width() // 3 // self.dpi,
                frame.winfo_height() // self.dpi // 3,
            )
        )
        names = ["Прямоугольники", "Трапеции", "Симпсон"]
        values = abs(np.array(self.answer[0:-1]) - self.answer[-1])
        print(names, values)
        ax.bar(
            names,
            values,
            color=self.colors["mid"],
        )
        fig.suptitle(
            "Отклонение метода от точного значения", color=self.colors["orange"]
        )
        ax.set_yscale("log")

        fig.set_facecolor(self.colors["dark"])
        ax.set_facecolor(self.colors["dark"])
        ax.tick_params(axis="x", colors=self.colors["orange"])
        ax.tick_params(axis="y", colors=self.colors["orange"])
        ax.spines["right"].set_color(self.colors["orange"])
        ax.spines["bottom"].set_color(self.colors["orange"])

        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(
            side="top", expand=1, fill="both"
        )

        for element in self.destroyables:
            element.destroy()

    def makeagraph(self, a, b, eps, func, frame, ans, type):
        fig, ax = subplots(
            figsize=(
                frame.winfo_width() // 3 // self.dpi,
                frame.winfo_height() // self.dpi // 3,
            )
        )

        if type == "Bisect":
            X = np.arange(a, b, eps)
            fun = np.vectorize(func)
            Y = fun(X.astype(float))
            ax.plot(X, Y, color=self.colors["mid"])
            ax.scatter(ans, fun(ans), color="white")
        if type == "just_line":
            X = np.arange(a, b, eps)
            fun = np.vectorize(func)
            Y = fun(X.astype(float))
            ax.plot(X, Y)
        elif type == "FPI":
            X = np.arange(a, b, eps)
            fun = np.vectorize(func)
            Y = fun(X.astype(float))
            ax.plot(X, Y, color=self.colors["mid"])
            [
                ax.scatter(
                    answer,
                    fun(np.array(answer).astype(float)),
                    color=self.colors["orange"],
                )
                if len(answer) != 0
                else None
                for answer in ans
            ]
        elif type == "regress":
            X = np.arange(a, b, eps)
            lab = ["Метод Лагранжа", "Метод ньютона"]
            for function in func[0:-2]:
                func_copy = function
                function = np.vectorize(sympy.lambdify(sympy.symbols("x"), function))
                Y = function(X.astype(float))
                print(Y)
                print(self.answer)
                ax.plot(X, Y, label=func_copy)
            for i in range(2):
                function = np.vectorize(
                    sympy.lambdify(sympy.symbols("x"), func[-1 - i])
                )
                Y = function(X.astype(float))
                print(Y)
                print(self.answer)
                ax.plot(X, Y, "-" if i == 0 else "--", label=lab[i])
            ax.scatter(ans[0], ans[1], color=self.colors["orange"])
            ax.legend()
        elif type == "diff":
            X = np.arange(a, b + eps / 10, eps / 10)
            function = np.vectorize(sympy.lambdify(sympy.symbols("x"), func[-1]))
            Y = function(X.astype(float))
            ax.plot(X, Y, linewidth=4, label="Аналитическое решение")
            ax.plot(func[0][0], func[0][1], "--o", label="Метод Эйлера")
            ax.plot(func[1][0], func[1][1], "--o", label="Метод Рунге-Кутты")
            ax.legend()

        fig.set_facecolor(self.colors["dark"])
        ax.set_facecolor(self.colors["dark"])
        ax.tick_params(axis="x", colors=self.colors["orange"])
        ax.tick_params(axis="y", colors=self.colors["orange"])
        ax.spines["right"].set_color(self.colors["orange"])
        ax.spines["bottom"].set_color(self.colors["orange"])

        ax.set_xlim(a - (a + b) / 10, b + (a + b) / 10)
        ax.grid(which="both")
        # ax.spines["left"].set_position("zero")
        ax.spines["left"].set_color("none")
        # ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        # ax.set_xticks(np.arange(a, b, (a - b) / 20))

        FigureCanvasTkAgg(fig, frame).get_tk_widget().pack(
            side="top", expand=1, fill="both"
        )
        print("we've reached the draw deleting")
        for element in self.destroyables:
            element.destroy()
        print("we've reached the draw return")
        return None


met = kursach.NumMethods()
a = interface(int(1980 // 1.5), int(1080 // 1.5), met)
