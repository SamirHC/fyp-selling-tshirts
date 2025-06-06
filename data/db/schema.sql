--DROP TABLE IF EXISTS "palettes";

CREATE TABLE IF NOT EXISTS "palettes" (
  "id" INTEGER,
  "likes" INTEGER,
  "submission_date" TEXT,
  "color_hunt_id" TEXT UNIQUE ON CONFLICT REPLACE,
  PRIMARY KEY ("id")
);

--DROP TABLE IF EXISTS "palette_tags";

CREATE TABLE IF NOT EXISTS "palette_tags" (
  "name" TEXT,
  "is_colour_tag" BOOLEAN,
  PRIMARY KEY ("name")
);

--DROP TABLE IF EXISTS "palette_tag_associations";

CREATE TABLE IF NOT EXISTS "palette_tag_associations" (
  "palette_id" INTEGER,
  "tag" TEXT,
  PRIMARY KEY ("palette_id", "tag")
  FOREIGN KEY ("tag") REFERENCES "palette_tags" ("name") ON DELETE CASCADE
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id") ON DELETE CASCADE
);

--DROP TABLE IF EXISTS "palette_colours";

CREATE TABLE IF NOT EXISTS "palette_colours" (
  "palette_id" INTEGER,
  "colour" CHAR(7),
  PRIMARY KEY ("palette_id", "colour")
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id") ON DELETE CASCADE
);

--DROP TABLE IF EXISTS "clothes";

CREATE TABLE IF NOT EXISTS "clothes" (
  "source" TEXT,
  "item_id" TEXT,
  "title" TEXT,
  "image_url" TEXT UNIQUE,
  PRIMARY KEY ("source", "item_id")
);

--DROP TABLE IF EXISTS "print_design_palettes";

CREATE TABLE IF NOT EXISTS "print_design_palettes" (
  "source" TEXT,
  "item_id" TEXT,
  "colour" CHAR(7),
  PRIMARY KEY ("source", "item_id", "colour")
  FOREIGN KEY ("item_id", "source") REFERENCES "clothes" ("item_id", "source") ON DELETE CASCADE
);

--DROP TABLE IF EXISTS "print_design_regions";

CREATE TABLE IF NOT EXISTS "print_design_regions" (
  "source" TEXT,
  "item_id" TEXT,
  "algorithm" TEXT,
  "left" INTEGER,
  "top" INTEGER,
  "width" INTEGER,
  "height" INTEGER,
  PRIMARY KEY ("source", "item_id", "algorithm")
  FOREIGN KEY ("item_id", "source") REFERENCES "clothes" ("item_id", "source") ON DELETE CASCADE
);

--DROP TABLE IF EXISTS "print_design_tags";

CREATE TABLE IF NOT EXISTS "print_design_tags" (
  "source" TEXT,
  "design_id" TEXT,
  "tag" TEXT,
  PRIMARY KEY ("source", "design_id", "tag")
  FOREIGN KEY ("design_id", "source") REFERENCES "clothes" ("item_id", "source") ON DELETE CASCADE
);

--DROP TABLE IF EXISTS "print_design_nearest_palette";

CREATE TABLE IF NOT EXISTS "print_design_nearest_palette" (
  "source" TEXT,
  "design_id" TEXT,
  "palette_id" INTEGER,
  "distance" FLOAT,
  PRIMARY KEY ("source", "design_id")
  FOREIGN KEY ("design_id", "source") REFERENCES "clothes" ("item_id", "source") ON DELETE CASCADE
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id") ON DELETE CASCADE
);

--DROP TABLE IF EXISTS "fonts";

CREATE TABLE IF NOT EXISTS "fonts" (
  "filename" TEXT,
  "base_font_name" TEXT,
  "file_format" TEXT,
  "creator" TEXT,
  "category" TEXT,
  "theme" TEXT,
  PRIMARY KEY ("filename")
);

-- Evaluation Tables
--DROP TABLE IF EXISTS "evaluate_palette_tag_matching";

CREATE TABLE IF NOT EXISTS "evaluate_palette_tag_matching" (
	"source"	TEXT,
	"item_id"	TEXT,
	"colour_match_count"	INTEGER,
  "tag_tp" INTEGER,
  "tag_fp" INTEGER,
	PRIMARY KEY("source","item_id"),
	FOREIGN KEY("source", "item_id") REFERENCES "clothes"("source", "item_id")
);
