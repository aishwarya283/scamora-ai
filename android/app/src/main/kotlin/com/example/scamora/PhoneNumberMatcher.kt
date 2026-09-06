package com.example.scamora

import android.telephony.PhoneNumberUtils

object PhoneNumberMatcher {
    /**
     * Normalize phone numbers to a canonical digits-only representation.
     * Rules:
     *  - Remove all non-digit characters
     *  - If digits >= 10, return the last 10 digits (heuristic for many national numbers)
     */
    @JvmStatic
    fun normalize(number: String?): String {
        if (number == null) return ""
        var s = number.trim()
        if (s.isEmpty()) return ""

        s = PhoneNumberUtils.normalizeNumber(s)
        val digits = s.replace(Regex("[^0-9]"), "")
        if (digits.isEmpty()) return ""

        val canonical = when {
            digits.length == 10 -> digits
            digits.length > 10 -> digits.takeLast(10)
            else -> digits
        }

        return canonical
    }

    @JvmStatic
    fun areSame(a: String?, b: String?): Boolean {
        val na = normalize(a)
        val nb = normalize(b)
        if (na.isEmpty() || nb.isEmpty()) return false
        return na == nb
    }
}
