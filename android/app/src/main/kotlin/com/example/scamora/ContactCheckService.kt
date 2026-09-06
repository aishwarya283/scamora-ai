package com.example.scamora

import android.Manifest
import android.app.Activity
import android.content.ContentResolver
import android.content.pm.PackageManager
import android.database.Cursor
import android.provider.ContactsContract
import android.telephony.PhoneNumberUtils
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject

class ContactCheckService(private val activity: Activity, private val channel: MethodChannel) {

    companion object {
        const val METHOD_NORMALIZE = "normalizePhoneNumber"
        const val METHOD_IS_KNOWN = "isKnownContact"
        const val PERMISSION_REQUEST_CODE = 9843
    }

    fun register() {
        channel.setMethodCallHandler { call, result ->
            when (call.method) {
                METHOD_NORMALIZE -> {
                    val input = call.argument<String>("number") ?: ""
                    result.success(PhoneNumberMatcher.normalize(input))
                }
                METHOD_IS_KNOWN -> {
                    val number = call.argument<String>("number") ?: ""
                    val response = isKnownContact(number)
                    result.success(response.toString())
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun jsonResult(known: Boolean, reason: String): JSONObject {
        val o = JSONObject()
        o.put("known", known)
        o.put("reason", reason)
        return o
    }

    fun isKnownContact(rawNumber: String): JSONObject {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(activity, arrayOf(Manifest.permission.READ_CONTACTS), PERMISSION_REQUEST_CODE)
            return jsonResult(false, "permission_denied")
        }

        val resolver: ContentResolver = activity.contentResolver
        val normalizedQuery = PhoneNumberMatcher.normalize(rawNumber)
        if (normalizedQuery.isEmpty()) return jsonResult(false, "no_match")

        val projection = arrayOf(ContactsContract.CommonDataKinds.Phone.NUMBER)
        var cursor: Cursor? = null
        try {
            cursor = resolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                null,
                null,
                null
            )
            if (cursor == null) return jsonResult(false, "no_contacts")
            while (cursor.moveToNext()) {
                val contactNumber = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                val normalizedContact = PhoneNumberMatcher.normalize(contactNumber)
                if (normalizedContact.isNotEmpty() && normalizedContact == normalizedQuery) {
                    return jsonResult(true, "match_found")
                }
                if (PhoneNumberUtils.compare(normalizedContact, normalizedQuery)) {
                    return jsonResult(true, "match_found")
                }
            }
            return jsonResult(false, "no_match")
        } catch (ex: Exception) {
            ex.printStackTrace()
            return jsonResult(false, "no_match")
        } finally {
            cursor?.close()
        }
    }
}
