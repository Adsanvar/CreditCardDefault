default = data[data['default payment next month'] == 1]

# objects = ('default', 'no_default')
# y_pos = [0, 1]
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.bar(0, len(default), align ='center', alpha=.4, label ="Default: " + str(len(default)) )
# #ax.bar(1, len(n_default), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(n_default)))
# fig.legend()
# plt.xticks(y_pos, objects)
# #fig.savefig('imbalanced_data.png', format='png')
# plt.show()