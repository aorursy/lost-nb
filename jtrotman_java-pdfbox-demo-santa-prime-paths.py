#!/usr/bin/env python
# coding: utf-8



from IPython.display import IFrame
from glob import glob




get_ipython().run_cell_magic('writefile', 'RenderSantaTripToPDF.java', 'import java.awt.Color;\nimport java.io.BufferedReader;\nimport java.io.File;\nimport java.io.FileReader;\nimport java.io.IOException;\nimport java.io.PrintStream;\nimport java.util.ArrayList;\n\nimport org.apache.pdfbox.pdmodel.PDDocument;\nimport org.apache.pdfbox.pdmodel.PDPage;\nimport org.apache.pdfbox.pdmodel.PDPageContentStream;\nimport org.apache.pdfbox.pdmodel.common.PDRectangle;\n\npublic class RenderSantaTripToPDF {\n\n\tpublic static void main(String[] args) throws Exception {\n\n\t\tif (args.length < 2) {\n\t\t\tSystem.out.println("Usage: java RenderSantaTripToPDF cities.csv solution.csv [output.pdf] [-c]");\n\t\t\treturn;\n\t\t}\n\n\t\tfinal String cities_csv = args[0];\n\t\tfinal String sol_csv = args[1];\n\t\tfinal String out_pdf = (args.length > 2 ? args[2] : null);\n\t\tfinal boolean hsvColor = (args.length > 3 && args[3].startsWith("-c"));\n\n\t\tRenderSantaTripToPDF s = new RenderSantaTripToPDF();\n\t\ts.readCities(cities_csv);\n\t\ts.drawSolution(sol_csv, hsvColor);\n\t\ts.saveAndClose(out_pdf);\n\t\ts.printStats(System.out);\n\t}\n\n\tstatic boolean isPrime(final int n) {\n\t\t// check if n is a multiple of 2\n\t\tif (n % 2 == 0)\n\t\t\treturn false;\n\t\t// if not, then just check the odds\n\t\tfor (int i = 3; i * i <= n; i += 2) {\n\t\t\tif (n % i == 0)\n\t\t\t\treturn false;\n\t\t}\n\t\treturn true;\n\t}\n\n\tArrayList<double[]> cities = new ArrayList<double[]>();\n\tPDDocument doc;\n\tPDPage page;\n\tPDPageContentStream contentStream;\n\n\t// from cities.csv\n\tfinal float maxX = 5100;\n\tfinal float maxY = 3400;\n\t// settings to tweak\n\tfloat border = 50;\n\tfloat lineWidth = 1.5f;\n\tfloat startMarkerSize = 5f; // diamond marking start of path, size in original units\n\t// computed score stats\n\tint nprime;\n\tint npenalised;\n\tdouble sum;\n\tdouble penalty;\n\n\tRenderSantaTripToPDF() throws Exception {\n\t\tdoc = new PDDocument();\n\t\tfloat expand = 2 * border;\n\t\tfinal PDRectangle size = new PDRectangle(maxX + expand, maxY + expand);\n\t\tpage = new PDPage(size); // replace PDPage.PAGE_SIZE_A4\n\t\tdoc.addPage(page);\n\t\tcontentStream = new PDPageContentStream(doc, page, true, true);\n\t\tcontentStream.setLineWidth(lineWidth);\n\t}\n\n\tvoid saveAndClose(String out_pdf) throws Exception {\n\t\tcontentStream.close();\n\t\tdoc.save(pdfName(out_pdf));\n\t\tdoc.close();\n\t}\n\n\tString pdfName(String out_pdf) {\n\t\treturn out_pdf != null ? out_pdf : String.format("%.2f.pdf", sum + penalty);\n\t}\n\n\tvoid printStats(PrintStream out) {\n\t\tout.println("TSP score       : " + sum);\n\t\tout.println("Prime penalty   : " + penalty);\n\t\tout.println("10ths prime     : " + nprime);\n\t\tout.println("10ths penalised : " + npenalised);\n\t\tout.println("FINAL SCORE     : " + (sum + penalty));\n\t}\n\n\tvoid readCities(String cities_csv) throws IOException {\n\n\t\tBufferedReader r = new BufferedReader(new FileReader(new File(cities_csv)));\n\t\tr.readLine(); // header\n\t\tString line;\n\n\t\twhile ((line = r.readLine()) != null) {\n\t\t\tfinal String[] p = line.split(",");\n\t\t\tfinal double x = Double.parseDouble(p[1]);\n\t\t\tfinal double y = Double.parseDouble(p[2]);\n\t\t\tcities.add(new double[] { x, y }); // use original coordinates in the page\n\t\t}\n\t\tr.close();\n\t}\n\n\tfloat pageX(double[] c) {\n\t\treturn (float) c[0] + border;\n\t}\n\n\tfloat pageY(double[] c) {\n\t\treturn (float) c[1] + border;\n\t}\n\n\tstatic double distance(double[] a, double[] b) {\n\t\tdouble dx = a[0] - b[0];\n\t\tdouble dy = a[1] - b[1];\n\t\treturn Math.sqrt((dx * dx) + (dy * dy));\n\t}\n\n\tvoid drawSolution(String sol_csv, boolean hsvColor) throws Exception {\n\n\t\tBufferedReader r = new BufferedReader(new FileReader(new File(sol_csv)));\n\t\tr.readLine(); // header\n\t\tr.readLine(); // city 0\n\n\t\tdouble[] depot = cities.get(0);\n\n\t\tcontentStream.setNonStrokingColor(Color.RED);\n\t\tfloat[] xs = { pageX(depot) - startMarkerSize, pageX(depot), pageX(depot) + startMarkerSize, pageX(depot) };\n\t\tfloat[] ys = { pageY(depot), pageY(depot) - startMarkerSize, pageY(depot), pageY(depot) + startMarkerSize };\n\t\tcontentStream.fillPolygon(xs, ys);\n\n\t\tString line;\n\t\tint ind = 1;\n\t\tint lastCity = 0;\n\t\tdouble[] last = cities.get(lastCity);\n\t\twhile ((line = r.readLine()) != null) {\n\t\t\tfinal int city = Integer.parseInt(line);\n\t\t\tdouble[] curr = cities.get(city);\n\t\t\tdouble link_score = distance(curr, last);\n\t\t\tsum += link_score;\n\n\t\t\tColor c = Color.black;\n\t\t\tif ((ind % 10) == 0) {\n\t\t\t\tif (isPrime(lastCity)) {\n\t\t\t\t\tc = Color.blue;\n\t\t\t\t\tnprime++;\n\t\t\t\t}\n\t\t\t\telse {\n\t\t\t\t\tc = Color.red;\n\t\t\t\t\tnpenalised++;\n\t\t\t\t\tpenalty += link_score * 0.1;\n\t\t\t\t}\n\t\t\t}\n\n\t\t\tif (hsvColor) {\n\t\t\t\t// simple alternate color scheme\n\t\t\t\tc = new Color(Color.HSBtoRGB(ind / (float) cities.size(), 1f, 1f));\n\t\t\t}\n\n\t\t\tcontentStream.setStrokingColor(c);\n\t\t\tcontentStream.drawLine(pageX(last), pageY(last), pageX(curr), pageY(curr));\n\n\t\t\tind++;\n\t\t\tlast = curr;\n\t\t\tlastCity = city;\n\t\t}\n\t\tr.close();\n\t}\n}')




cpath = '../input/java-jars/pdfbox-app-2.0.13.jar'
cities = '../input/traveling-santa-2018-prime-paths/cities.csv'




get_ipython().system('javac -cp {cpath} RenderSantaTripToPDF.java')




get_ipython().system('ls -l')




get_ipython().system('java -cp .:{cpath} RenderSantaTripToPDF {cities} ../input/pmtest1/submission.csv')




get_ipython().system('ls -l')




IFrame("1514325.09.pdf", width=900, height=650)




glob('../input/**/*.csv', recursive=True)

