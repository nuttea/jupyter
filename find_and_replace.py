#created by Jonathan Malott (jm72636) on Sep 19 2016
from tempfile import NamedTemporaryFile
import shutil
import csv

#CSVs in a csv directory
macros = ["Models.csv"]

for x in macros:
    filename = 'edmunds_try.csv'
    tempfile = NamedTemporaryFile(delete=False)

    with open(filename, 'rb') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"')
        writer = csv.writer(tempfile, delimiter=',', quotechar='"')

        for row in reader:

            #this item is the forum post
            item = row[2]

            with open('csv/'+x, 'rb') as csvfile:
                read = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row2 in read:
                        #replace
                        row[2] = row[2].lower().replace(row2[1].lower(),row2[0].lower())

            writer.writerow(row)

    shutil.move(tempfile.name, filename)
    print "Working on "+x
