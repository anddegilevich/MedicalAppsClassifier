import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os

class Main(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.init_main()
        self.db = db
        self.view_records()

    def init_main(self):
        toolbar = tk.Frame(bg='#F4F4F6', bd=2)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.add_img = tk.PhotoImage(file="add.gif")
        btn_open_dialog = tk.Button(toolbar, text='Добавить', command=self.open_dialog, bg='#F4F4F6', bd=0,
                                    compound=tk.TOP, image=self.add_img)
        btn_open_dialog.pack(side=tk.LEFT)

        self.update_img = tk.PhotoImage(file="update.gif")
        btn_edit_dialog = tk.Button(toolbar, text='Редактировать', bg='#F4F4F6', bd=0, image=self.update_img,
                                    compound=tk.TOP, command=self.open_update_dialog)
        btn_edit_dialog.pack(side=tk.LEFT)

        self.delete_img = tk.PhotoImage(file="delete.gif")
        btn_delete = tk.Button(toolbar, text='Удалить', bg='#F4F4F6', bd=0, image=self.delete_img,
                                    compound=tk.TOP, command=self.delete_records)
        btn_delete.pack(side=tk.LEFT)

        self.search_img = tk.PhotoImage(file="search.gif")
        btn_search = tk.Button(toolbar, text='Поиск', bg='#F4F4F6', bd=0, image=self.search_img,
                               compound=tk.TOP, command=self.open_search_dialog)
        btn_search.pack(side=tk.LEFT)

        self.refresh_img = tk.PhotoImage(file="refresh.gif")
        btn_refresh = tk.Button(toolbar, text='Обновить', bg='#F4F4F6', bd=0, image=self.refresh_img,
                               compound=tk.TOP, command=self.view_records)
        btn_refresh.pack(side=tk.LEFT)

        self.sort_img = tk.PhotoImage(file="sort.gif")
        btn_sort = tk.Button(toolbar, text='Сортировать', bg='#F4F4F6', bd=0, image=self.sort_img,
                                compound=tk.TOP, command=self.open_sort_dialog)
        btn_sort.pack(side=tk.LEFT)

        self.analysis_img = tk.PhotoImage(file="analysis.gif")
        btn_analysis = tk.Button(toolbar, text='Анализ', bg='#F4F4F6', bd=0, image=self.analysis_img,
                             compound=tk.TOP, command=self.open_analysis_dialog)
        btn_analysis.pack(side=tk.LEFT)

        self.info_img = tk.PhotoImage(file="info.gif")
        btn_info = tk.Button(toolbar, text='Справка', bg='#F4F4F6', bd=0, image=self.info_img,
                                 compound=tk.TOP, command=self.open_info_dialog)
        btn_info.pack(side=tk.LEFT)

        self.BackGround = tk.PhotoImage(file="BackGround.gif")
        canvas = Canvas(toolbar, width=670, height=50, bg='#F4F4F6')
        canvas.create_image(0, 0, anchor=NW, image=self.BackGround)
        canvas.pack(side=tk.LEFT)


        column_names = ['ID', 'name', 'category', 'Type', 'Disease', 'rating', 'size', 'installs']
        column_text = ['ID', 'Название', 'Категория', 'Тип приложения', 'Болезнь', 'Рейтинг', 'Размер', 'Количество скачиваний']
        column_width = [40, 220, 220, 220, 220, 100, 100, 150]
        column_N = len(column_names)

        self.tree = ttk.Treeview(self, columns=(column_names),
                                 height=29, show='headings')

        for i in range(column_N):
            self.tree.column(column_names[i], width=column_width[i], anchor=tk.CENTER)
            self.tree.heading(column_names[i], text=column_text[i])

        self.tree.pack(side=tk.LEFT)

        scroll = tk.Scrollbar(self, command=self.tree.yview)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)


    def records(self, name, category, Type, Disease, rating, size, installs):
        self.db.insert_data(name, category, Type, Disease, rating, size, installs)
        self.view_records()

    def update_record(self, name, category, Type, Disease, rating, size, installs):
        self.db.c.execute('''UPDATE data SET name=?, category=?, Type=?, Disease=?, rating=?, size=?, installs=? WHERE ID=?''',
                          (name, category, Type, Disease, rating, size, installs, self.tree.set(self.tree.selection()[0], '#1')))
        self.db.conn.commit()
        self.view_records()

    def delete_records(self):
        for selection_item in self.tree.selection():
            self.db.c.execute('''DELETE FROM data WHERE ID=?''', (self.tree.set(selection_item, '#1'),))
        self.db.conn.commit()
        self.view_records()

    def search_records(self, name):
        name = ('%' + name + '%',)
        self.db.c.execute('''SELECT * FROM data WHERE name LIKE ?''', name)
        [self.tree.delete(i) for i in self.tree.get_children()]
        [self.tree.insert('', 'end', values = row) for row in self.db.c.fetchall()]

    def sort_records(self, columnname, by):
        self.db.c.execute('SELECT * FROM data ORDER BY ' + columnname + ' ' + by)
        [self.tree.delete(i) for i in self.tree.get_children()]
        [self.tree.insert('', 'end', values = row) for row in self.db.c.fetchall()]

    def view_records(self):
        self.db.c.execute('''SELECT * FROM data''')
        [self.tree.delete(i) for i in self.tree.get_children()]
        [self.tree.insert('', 'end', values = row) for row in self.db.c.fetchall()]

    def open_dialog(self):
        Child()

    def open_update_dialog(self):
        if self.tree.selection() == ():
            self.ChooseApp()
            return
        Update()

    def open_search_dialog(self):
        Search()

    def open_sort_dialog(self):
        Sort()

    def open_analysis_dialog(self):
        if self.tree.selection() == ():
            self.ChooseApp()
            return
        Analysis()

    def open_info_dialog(self):
        Info()

    def ChooseApp(self):
        messagebox.showinfo('Ошибка', 'Выберите приложение')


class Child(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.init_child()
        self.view = app

    def init_child(self):
        self.title('Добавить приложение')
        self.geometry('320x280+640+360')
        self.resizable(False, False)

        label_name = tk.Label(self, text='Название:')
        label_name.place(x=20, y=20)
        label_category = tk.Label(self, text='Категория:')
        label_category.place(x=20, y=50)
        label_Type = tk.Label(self, text='Тип приложения:')
        label_Type.place(x=20, y=80)
        label_Disease = tk.Label(self, text='Болезнь:')
        label_Disease.place(x=20, y=110)
        label_rating = tk.Label(self, text='Рейтинг:')
        label_rating.place(x=20, y=140)
        label_size = tk.Label(self, text='Размер:')
        label_size.place(x=20, y=170)
        label_installs = tk.Label(self, text='Установки:')
        label_installs.place(x=20, y=200)

        self.entry_name = ttk.Entry(self)
        self.entry_name.place(x=150, y=20)

        self.combobox_category = ttk.Combobox(self, values=[u"Health&Care", u"Medical"])
        self.combobox_category.current(0)
        self.combobox_category.place(x=150, y=50)

        self.Types_text = ["Новости", "Информация", "Обучающий материал", "Проигрыватель", "Брокер",
                      "Поддержка принятия решений", "Калькулятор", "Измерительный прибор", "Монитор", "Трекер",
                      "Административные задачи", "Дневник", "Напоминание", "Календарь", "Помощь", "Тренер",
                      "Менеджер по здоровью", "Привод", "Коммуникатор", "Игра", "Магазин", "Прочее"]
        self.combobox_Type = ttk.Combobox(self, values=self.Types_text)
        self.combobox_Type.current(0)
        self.combobox_Type.place(x=150, y=80)

        self.Disease_text = ["Ожирение", "Женское здоровье", "Мышечная слабость", "Болезнь сердца",
                             "Психологическое заболевание", "Онкологическое заболевание", "Болезнь ротовой полости",
                             "Болезнь костей и суставов", "Диабет", "Болезнь Печени", "Комплексное заболевание",
                             "Болезнь животных", "Другое"]
        self.combobox_Disease = ttk.Combobox(self, values=self.Disease_text)
        self.combobox_Disease.current(0)
        self.combobox_Disease.place(x=150, y=110)

        self.entry_rating = ttk.Entry(self)
        self.entry_rating.place(x=150, y=140)

        self.entry_size = ttk.Entry(self)
        self.entry_size.place(x=150, y=170)

        self.entry_installs = ttk.Entry(self)
        self.entry_installs.place(x=150, y=200)

        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.place(x=50, y=230)

        self.btn_ok = ttk.Button(self, text='Добавить')
        self.btn_ok.place(x=180, y=230)
        self.btn_ok.bind('<Button-1>', lambda event: self.view.records(self.entry_name.get(),
                                                                  self.combobox_category.get(),
                                                                  self.combobox_Type.get(),
                                                                  self.combobox_Disease.get(),
                                                                  self.entry_rating.get(),
                                                                  self.entry_size.get(),
                                                                  self.entry_installs.get()))
        self.btn_ok.bind('<Button-1>', lambda event: self.destroy(), add='+')

        self.grab_set()
        self.focus_set()

class Update(Child):
    def __init__(self):
        super().__init__()
        self.init_edit()
        self.view = app
        self.db = db
        self.default_data()

    def init_edit(self):
        self.title('Редактирование')
        btn_edit = ttk.Button(self, text='Редактировать')
        btn_edit.place(x=180, y=230)
        btn_edit.bind('<Button-1>', lambda event: self.view.update_record(self.entry_name.get(),
                                                                  self.combobox_category.get(),
                                                                  self.combobox_Type.get(),
                                                                  self.combobox_Disease.get(),
                                                                  self.entry_rating.get(),
                                                                  self.entry_size.get(),
                                                                  self.entry_installs.get()))
        btn_edit.bind('<Button-1>', lambda event: self.destroy(), add='+')
        self.btn_ok.destroy()
        self.grab_set()
        self.focus_set()

    def default_data(self):
        self.db.c.execute('''SELECT * FROM data WHERE id=?''',
                          (self.view.tree.set(self.view.tree.selection()[0], '#1' ),))
        row = self.db.c.fetchone()

        self.entry_name.insert(0, row[1])

        if row[2] == 'Medical':
            self.combobox_category.current(1)


        Types_N = len(self.Types_text)
        for i in range(Types_N):
            if row[3] == self.Types_text[i]:
                self.combobox_Type.current(i)

        Disease_N = len(self.Disease_text)
        for i in range(Disease_N):
            if row[4] == self.Disease_text[i]:
                self.combobox_Disease.current(i)

        self.entry_rating.insert(0, row[5])

        self.entry_size.insert(0, row[6])

        self.entry_installs.insert(0, row[7])


class Search(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.init_search()
        self.view = app

    def init_search(self):
        self.title('Поиск')
        self.geometry('300x100+640+360')
        self.resizable(False, False)

        label_search = tk.Label(self, text='Поиск')
        label_search.place(x=50, y=20)

        self.entry_search = ttk.Entry(self)
        self.entry_search.place(x=105, y=20, width=150)

        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.place(x=185,y=50)

        btn_search = ttk.Button(self, text='Поиск')
        btn_search.place(x=105, y=50)
        btn_search.bind('<Button-1>', lambda event: self.view.search_records(self.entry_search.get()))
        btn_search.bind('<Button-1>', lambda event: self.destroy(), add='+')

        self.grab_set()
        self.focus_set()

class Sort(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.init_sort()
        self.view = app

    def init_sort(self):
        self.title('Сортировка')
        self.geometry('300x120+640+330')
        self.resizable(False, False)

        label_sort = tk.Label(self, text='Сортировать по:')
        label_sort.place(x=20, y=20)

        column_names = ['ID', 'name', 'category', 'Type', 'Disease', 'rating', 'size', 'installs']

        self.combobox_sort = ttk.Combobox(self, values=column_names)
        self.combobox_sort.current(0)
        self.combobox_sort.place(x=120, y=20, width=150)

        self.combobox_sortby = ttk.Combobox(self, values=['ASC', 'DESC'])
        self.combobox_sortby.current(0)
        self.combobox_sortby.place(x=120, y=50, width=150)

        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.place(x=200, y=80)

        btn_sort = ttk.Button(self, text='Сортировать')
        btn_sort.place(x=110, y=80)
        btn_sort.bind('<Button-1>', lambda event: self.view.sort_records(self.combobox_sort.get(),
                                                                         self.combobox_sortby.get()))
        btn_sort.bind('<Button-1>', lambda event: self.destroy(), add='+')

        self.grab_set()
        self.focus_set()

class Analysis(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.init_analysis()
        self.view = app
        self.db = db
        self.load_data()

    def init_analysis(self):
        self.title('Анализ')
        self.geometry('1300x720+200+150')
        self.resizable(False, False)

        self.text = Text(self, width=50, wrap=WORD, font=("Arial", 12), padx=10, pady=10)

        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.pack(side=RIGHT, padx=1, pady=2)

        self.grab_set()
        self.focus_set()

    def load_data(self):
        self.db.c.execute('''SELECT * FROM data WHERE id=?''',
                          (self.view.tree.set(self.view.tree.selection()[0], '#1'),))
        row = self.db.c.fetchone()
        name = row[1]
        category = row[2]
        Type = row[3]
        Disease = row[4]
        Rating = row[5]
        Size = row[6]
        Installs = row[7]
        Marks = row[9]
        Android = row[15]
        Content = row[16]
        Interactive_elements = row[17]
        Description = row[8]
        text_0 = '''Название приложения: {name}\n\nКатегория: {category}\nТип медицинского приложения: {Type}
Тип болезни: {Disease}\nРейтинг приложения: {Rating}\nРазмер приложения: {Size} МБайт\nКоличество скачиваний: {Installs}
Количество оценок: {Marks}\nВерсия Android: {Android}\nВозрастной рейтинг: {Content}
Требования к доступу: {Interactive_elements}\n\nОписание приложения: {Description}'''
        self.text.insert(END, text_0.format(name=name, category=category, Type=Type, Disease=Disease, Rating=Rating,
                                       Size=Size, Installs=Installs, Marks=Marks, Android=Android, Content=Content,
                                       Interactive_elements=Interactive_elements, Description=Description))

        self.text.config(state=DISABLED)
        self.text.pack(side=LEFT, fill=Y)

        scroll = tk.Scrollbar(self, command=self.text.yview)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.text.configure(yscrollcommand=scroll.set)

        files = os.listdir(path="../data")
        N = len(files)

        ratingY = np.zeros(N)
        Month = np.zeros(N)
        for i in range(N):
            if row[2] == 'Health&Care':
                address = '../data/' + files[i] + '/google_play_health-fitness.db'
            else:
                address = '../data/' + files[i] + '/google_play_medical.db'
            con = sqlite3.connect(address)
            df = pd.read_sql("SELECT * FROM apps", con)
            ratingY[i] = pd.to_numeric(df['rating'].loc[df['name'] == row[1]], errors='coerce')
            Month[i] = files[i]

        address = 'data.db'
        con = sqlite3.connect(address)
        df = pd.read_sql("SELECT * FROM data", con)
        class_df = df.loc[df['Type'] == row[3]]

        frame = Frame(self)
        frame.pack(fill=BOTH)

        figure = plt.Figure(figsize=(5,10))
        figure.subplots_adjust(hspace=0.5)
        ax1 = figure.add_subplot(3, 1, 1)
        bar = FigureCanvasTkAgg(figure, frame)
        bar.get_tk_widget().pack(side=TOP, fill=X)
        class_df['rating'].plot(kind='hist', bins=50, ax=ax1, xlim=[1,5])
        class_df['rating'].loc[class_df['rating'] == row[5]].plot(kind='hist', ax=ax1, bins=50, xlim=[1,5])
        ax1.set_title('Распределение рейтинга внутри класса', font="Arial", size=12)
        ax1.set_xlabel('Рейтинг', font="Arial", size=12)
        ax1.set_ylabel('Частота', font="Arial", size=12)
        ax1.grid()

        ax2 = figure.add_subplot(3, 1, 2)
        class_df['marks'].plot(kind='hist', bins=100, ax=ax2, xlim=[1,1000])
        ax2.plot([row[9], row[9]],[0, class_df.loc[class_df['marks'] == row[9]].shape[0]])
        ax2.set_title('Распределение количества оценок внутри класса', font="Arial", size=12)
        ax2.set_xlabel('Количество оценок', font="Arial", size=12)
        ax2.set_ylabel('Частота', font="Arial", size=12)
        ax2.grid()

        ax3 = figure.add_subplot(3, 1, 3)
        ax3.plot(np.arange(N),ratingY)
        ax3.set_xticks(np.arange(N))
        ax3.set_xticklabels(Month, rotation=45)
        ax3.set_title('Динамика рейтинга приложения', font="Arial", size=12)
        ax3.set_xlabel('Месяц', font="Arial", size=12)
        ax3.set_ylabel('Рейтинг', font="Arial", size=12)

class Info(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.init_info()
        self.view = app

    def init_info(self):
        self.title('Справка')
        self.geometry('1300x720+200+150')
        self.resizable(False, False)

        self.text = Text(self, wrap=WORD, font=("Arial", 20))
        f = open("Info.txt", encoding="utf-8")
        self.text.insert(1.0, f.read())
        self.text.config(state=DISABLED)
        self.text.pack(side=TOP, fill=BOTH)

        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.pack(side=BOTTOM, padx=1, pady=2)

        self.grab_set()
        self.focus_set()

class DB:
    def __init__(self):
        self.conn = sqlite3.connect('data.db')
        self.c=self.conn.cursor()
        self.conn.commit()

    def insert_data(self, name, category, Type, Disease, rating, size, installs):
        self.c.execute('''INSERT INTO data(name, category, Type, Disease, rating, size, installs) 
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                       (name, category, Type, Disease, rating, size, installs))
        self.conn.commit()

if __name__ == "__main__":
    root = tk.Tk()
    db = DB()
    app = Main(root)
    app.pack()
    root.title("Приложения медицинского назначения")
    root.geometry("1300x720+200+150")
    root.resizable(False, False)
    root.mainloop()